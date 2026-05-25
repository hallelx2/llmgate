package llmgate

import (
	"context"
	"sync"
	"sync/atomic"
)

// Mock is an in-memory Client for tests.
//
// It records every Request it sees, and responds with either a fixed
// Reply string or — if set — a function that computes the response
// from the incoming request. Calls, LastPrompts, and LastMessages are
// safe to read concurrently.
type Mock struct {
	// Reply is returned verbatim as Response.Content when Respond is nil.
	Reply string

	// Respond, when non-nil, is called to build the Response. It
	// overrides Reply.
	Respond func(ctx context.Context, req Request) (*Response, error)

	// TokensPerCall is the fixed token count returned by CountTokens when
	// CountFn is nil. Zero => approximate as len(text)/4.
	TokensPerCall int

	// CountFn, when non-nil, overrides CountTokens.
	CountFn func(ctx context.Context, text string) (int, error)

	calls        int32
	mu           sync.Mutex
	lastPrompts  []string    // backward compat: user messages only
	lastMessages [][]Message // all messages per call
}

// Complete records the request and returns the canned reply.
func (m *Mock) Complete(ctx context.Context, req Request) (*Response, error) {
	atomic.AddInt32(&m.calls, 1)

	// Record the last user message for backward-compat assertions.
	var last string
	for _, msg := range req.Messages {
		if msg.Role == RoleUser {
			last = msg.Content
		}
	}

	// Also record the full message list for richer assertions.
	msgCopy := make([]Message, len(req.Messages))
	copy(msgCopy, req.Messages)

	m.mu.Lock()
	m.lastPrompts = append(m.lastPrompts, last)
	m.lastMessages = append(m.lastMessages, msgCopy)
	m.mu.Unlock()

	if m.Respond != nil {
		return m.Respond(ctx, req)
	}

	inTok := len(last) / 4
	outTok := len(m.Reply) / 4
	return &Response{
		Content:      m.Reply,
		InputTokens:  inTok,
		OutputTokens: outTok,
		Model:        req.Model,
		FinishReason: "stop",
		Usage: Usage{
			InputTokens:  inTok,
			OutputTokens: outTok,
			TotalTokens:  inTok + outTok,
		},
	}, nil
}

// CountTokens returns TokensPerCall, CountFn, or a length-based estimate.
func (m *Mock) CountTokens(ctx context.Context, text string) (int, error) {
	if m.CountFn != nil {
		return m.CountFn(ctx, text)
	}
	if m.TokensPerCall > 0 {
		return m.TokensPerCall, nil
	}
	return len(text) / 4, nil
}

// Calls returns the number of Complete invocations.
func (m *Mock) Calls() int { return int(atomic.LoadInt32(&m.calls)) }

// LastPrompts returns a copy of every user prompt seen so far.
// For richer assertions (including system/assistant messages), use
// LastMessages instead.
func (m *Mock) LastPrompts() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]string, len(m.lastPrompts))
	copy(out, m.lastPrompts)
	return out
}

// LastMessages returns a copy of the full message list from every
// Complete call. Each element is the []Message from one call.
// This lets tests assert on system prompts and multi-turn context.
func (m *Mock) LastMessages() [][]Message {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([][]Message, len(m.lastMessages))
	for i, msgs := range m.lastMessages {
		cp := make([]Message, len(msgs))
		copy(cp, msgs)
		out[i] = cp
	}
	return out
}
