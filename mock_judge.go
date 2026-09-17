package llmgate

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
)

// MockJudge is an in-memory Judge for tests, mirroring Mock.
//
// It records every JudgeRequest it sees and answers from canned values, a
// response function, or — when neither is set — a default that is
// well-formed for every question in the request.
//
// That default matters more than it sounds. A mock that answered with an
// empty distribution would let a caller's thresholding logic pass without
// ever running: every probability would be zero, every branch would take
// the same path, and the test would prove nothing. So the default builds a
// real distribution that sums to 1 for each question, from the question's
// own shape.
type MockJudge struct {
	// Answers is the canned answer per question ID. IDs present here are
	// returned verbatim; IDs absent from it fall back to the default.
	Answers map[string]Answer

	// Respond, when non-nil, builds the whole Judgment and overrides
	// Answers.
	Respond func(ctx context.Context, req JudgeRequest) (*Judgment, error)

	// Err, when non-nil, is returned instead of a judgment. Set it to
	// exercise a caller's failure path.
	Err error

	// Model is reported on the Judgment. Empty reports the request's model,
	// which mimics a provider resolving an alias to itself.
	Model string

	// SkipValidation disables the local request check.
	//
	// The mock validates by default so a test cannot pass with a question
	// the real transport would refuse to send. Turn it off only to exercise
	// behaviour on a deliberately malformed request.
	SkipValidation bool

	calls    int32
	mu       sync.Mutex
	requests []JudgeRequest
}

// Compile-time check that the mock satisfies the seam.
var _ Judge = (*MockJudge)(nil)

// ErrMockContextLength is returned by MockJudge when OverContextLimit is
// set on the request it receives via SimulateContextLimit.
var ErrMockContextLength = &LLMError{
	Class:    ErrClassContextLength,
	Provider: "mock",
	Message:  "simulated context-limit rejection",
}

// SimulateContextLimit makes the next and all later calls fail the way a
// transport's context guard does, without any network.
//
// Worth exercising: the guard is the one failure a caller hits by sending
// too many questions at once, and the fix — batching — is a code path most
// tests never reach otherwise.
func (m *MockJudge) SimulateContextLimit() { m.Err = ErrMockContextLength }

// Judge records the request and returns the canned or default judgment.
func (m *MockJudge) Judge(ctx context.Context, req JudgeRequest) (*Judgment, error) {
	atomic.AddInt32(&m.calls, 1)

	m.mu.Lock()
	m.requests = append(m.requests, recordRequest(req))
	m.mu.Unlock()

	if !m.SkipValidation {
		if err := req.Validate(); err != nil {
			return nil, err
		}
	}
	if m.Err != nil {
		return nil, m.Err
	}
	if m.Respond != nil {
		return m.Respond(ctx, req)
	}

	answers := make(map[string]Answer, len(req.Questions))
	for id, q := range req.Questions {
		if canned, ok := m.Answers[id]; ok && canned != nil {
			answers[id] = canned
			continue
		}
		a, err := defaultAnswerFor(q)
		if err != nil {
			return nil, fmt.Errorf("mock judge: question %q: %w", id, err)
		}
		answers[id] = a
	}

	model := m.Model
	if model == "" {
		model = req.Model
	}

	return &Judgment{
		Model:   model,
		Answers: answers,
		Usage: Usage{
			InputTokens:    1,
			TotalTokens:    1,
			TokensReported: true,
		},
	}, nil
}

// recordRequest snapshots the question map so a later mutation by the
// caller cannot rewrite history.
//
// Test code reuses request structs, and a recorded request that changes
// after the call makes an assertion describe the wrong moment — the kind of
// failure that reads as a flake. The questions themselves are value types,
// so copying the map is enough to pin what was asked; State is left as the
// caller's reference, since a mock cannot deep-copy an arbitrary any.
func recordRequest(req JudgeRequest) JudgeRequest {
	if req.Questions == nil {
		return req
	}
	qs := make(map[string]Question, len(req.Questions))
	for id, q := range req.Questions {
		qs[id] = q
	}
	req.Questions = qs
	return req
}

// defaultAnswerFor builds a well-formed answer for q, with a distribution
// that actually sums to 1 so a caller's confidence and threshold logic runs
// against realistic input rather than a degenerate all-zero map.
func defaultAnswerFor(q Question) (Answer, error) {
	// Pointer forms first. Validate accepts a non-nil *Noul (value
	// receivers put the methods in the pointer's method set too), so the
	// mock has to answer one rather than calling it an unknown kind — a
	// request the real transport would send must not fail here.
	switch v := q.(type) {
	case *Noul:
		if v == nil {
			return nil, fmt.Errorf("question is a nil *Noul")
		}
		return defaultAnswerFor(*v)
	case *Choice:
		if v == nil {
			return nil, fmt.Errorf("question is a nil *Choice")
		}
		return defaultAnswerFor(*v)
	case *Score:
		if v == nil {
			return nil, fmt.Errorf("question is a nil *Score")
		}
		return defaultAnswerFor(*v)
	}

	switch v := q.(type) {
	case Noul:
		return NoulAnswer{Noul: 0.5}, nil

	case Choice:
		if len(v.Options) == 0 {
			return nil, fmt.Errorf("choice has no options")
		}
		// Spread evenly, then give the first option the remainder so the
		// distribution sums to exactly 1 and has an unambiguous winner.
		probs := make(map[string]float64, len(v.Options))
		share := 1.0 / float64(len(v.Options))
		var assigned float64
		for _, opt := range v.Options[1:] {
			probs[opt.Name] = share
			assigned += share
		}
		probs[v.Options[0].Name] = 1 - assigned
		return ChoiceAnswer{
			Choice:        v.Options[0].Name,
			Probabilities: probs,
			Confidence:    probs[v.Options[0].Name],
		}, nil

	case Score:
		if len(v.Levels) == 0 {
			return nil, fmt.Errorf("score has no levels")
		}
		legend := make(map[int]string, len(v.Levels))
		probs := make(map[int]float64, len(v.Levels))
		share := 1.0 / float64(len(v.Levels))
		var assigned float64
		for i, lvl := range v.Levels {
			legend[i] = lvl
			if i > 0 {
				probs[i] = share
				assigned += share
			}
		}
		probs[0] = 1 - assigned

		// The score is the probability-weighted position, computed the way
		// the real model computes it, so a caller thresholding on it sees a
		// value with the right shape.
		var score float64
		for i, p := range probs {
			score += float64(i) * p
		}
		return ScoreAnswer{
			Score:         score,
			Legend:        legend,
			Probabilities: probs,
			Confidence:    probs[0],
		}, nil

	default:
		return nil, fmt.Errorf("unknown question kind %q", q.Kind())
	}
}

// Calls returns the number of Judge invocations.
//
// The assertion most worth writing with it is that a batch of N questions
// produced ONE call, which is the property the whole seam exists for.
func (m *MockJudge) Calls() int { return int(atomic.LoadInt32(&m.calls)) }

// Requests returns a copy of every JudgeRequest seen so far, so a test can
// assert on the questions that were actually asked.
func (m *MockJudge) Requests() []JudgeRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]JudgeRequest, len(m.requests))
	copy(out, m.requests)
	return out
}

// LastRequest returns the most recent request, and false when there is
// none.
func (m *MockJudge) LastRequest() (JudgeRequest, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.requests) == 0 {
		return JudgeRequest{}, false
	}
	return m.requests[len(m.requests)-1], true
}
