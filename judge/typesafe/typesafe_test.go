package typesafe_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/judge/typesafe"
)

// newTestJudge wires a Judge at a stub server, with the context guard's
// estimator replaced by a cheap deterministic one so tests do not depend on
// the tokenizer's exact output.
func newTestJudge(t *testing.T, h http.HandlerFunc) *typesafe.Judge {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)

	j, err := typesafe.New(typesafe.Config{
		APIKey:      "test-key",
		BaseURL:     srv.URL,
		CountTokens: func(s string) (int, error) { return len(s), nil },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return j
}

func batch() llmgate.JudgeRequest {
	return llmgate.JudgeRequest{
		State: "Help! My payouts have been failing for 3 days.",
		Questions: map[string]llmgate.Question{
			"is_urgent": llmgate.Noul{Instructions: "Does this convey urgency?"},
		},
	}
}

// The load-bearing property of the whole seam: a batch is ONE request.
func TestBatchIsASingleRequest(t *testing.T) {
	var calls int
	var gotBody []byte
	var gotAuth, gotContentType, gotPath string

	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		gotBody, _ = io.ReadAll(r.Body)
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")
		gotPath = r.URL.Path
		writeAnswers(w, `{"is_urgent":{"type":"noul","noul":0.92},
			"team":{"type":"choice","choice":"billing","probabilities":{"billing":0.9,"technical":0.1},"confidence":0.8},
			"frustration":{"type":"score","score":1.6,"legend":{"0":"Calm","1":"Angry"},"probabilities":{"0":0.4,"1":0.6},"confidence":0.7}}`)
	})

	req := llmgate.JudgeRequest{
		State: "state",
		Questions: map[string]llmgate.Question{
			"is_urgent": llmgate.Noul{Instructions: "urgent?"},
			"team": llmgate.Choice{Instructions: "which team?", Options: llmgate.ChoiceOptions{
				{Name: "billing"}, {Name: "technical"},
			}},
			"frustration": llmgate.Score{Instructions: "how frustrated?", Levels: []string{"Calm", "Angry"}},
		},
	}

	got, err := j.Judge(context.Background(), req)
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if calls != 1 {
		t.Errorf("made %d HTTP calls for a 3-question batch, want exactly 1", calls)
	}
	if gotPath != "/v1/systemone" {
		t.Errorf("path = %q, want /v1/systemone", gotPath)
	}
	if gotAuth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want Bearer test-key", gotAuth)
	}
	if gotContentType != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", gotContentType)
	}
	if len(got.Answers) != 3 {
		t.Errorf("got %d answers, want 3", len(got.Answers))
	}

	// The body must carry all three questions and the default model.
	var sent struct {
		State     string                     `json:"state"`
		Model     string                     `json:"model"`
		Questions map[string]json.RawMessage `json:"questions"`
	}
	if err := json.Unmarshal(gotBody, &sent); err != nil {
		t.Fatalf("request body was not valid JSON: %v", err)
	}
	if sent.State != "state" {
		t.Errorf("state = %q, want %q", sent.State, "state")
	}
	if sent.Model != typesafe.DefaultModel {
		t.Errorf("model = %q, want %q", sent.Model, typesafe.DefaultModel)
	}
	if len(sent.Questions) != 3 {
		t.Errorf("sent %d questions, want 3", len(sent.Questions))
	}
}

func TestDecodesAllThreePrimitives(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, `{
			"urgent":{"type":"noul","noul":0.92},
			"team":{"type":"choice","choice":"technical","probabilities":{"billing":0.08,"technical":0.85,"sales":0.07},"confidence":0.82},
			"frustration":{"type":"score","score":1.6,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0.05,"1":0.3,"2":0.65},"confidence":0.78}}`)
	})

	got, err := j.Judge(context.Background(), llmgate.JudgeRequest{
		State: "a ticket",
		Questions: map[string]llmgate.Question{
			"urgent": llmgate.Noul{Instructions: "urgent?"},
			"team": llmgate.Choice{Instructions: "which team?", Options: llmgate.ChoiceOptions{
				{Name: "billing"}, {Name: "technical"}, {Name: "sales"},
			}},
			"frustration": llmgate.Score{Instructions: "how frustrated?", Levels: []string{"Calm", "Frustrated", "Very angry"}},
		},
	})
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}

	if n, err := got.Noul("urgent"); err != nil || n != 0.92 {
		t.Errorf("Noul = %v, %v; want 0.92, nil", n, err)
	}

	c, err := got.Choice("team")
	if err != nil {
		t.Fatalf("Choice: %v", err)
	}
	if c.Choice != "technical" || c.Confidence != 0.82 || c.Probabilities["technical"] != 0.85 {
		t.Errorf("Choice = %+v, want technical/0.82/0.85", c)
	}

	s, err := got.Score("frustration")
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	// Level keys arrive as strings on the wire and must come back as ints,
	// so callers can order and index them meaningfully.
	if s.Score != 1.6 || s.Legend[2] != "Very angry" || s.Probabilities[2] != 0.65 {
		t.Errorf("Score = %+v, want 1.6 with int-keyed legend and distribution", s)
	}
}

// The resolved version, not the alias that was asked for. A threshold tuned
// against one version is not evidence about another, so the caller has to
// be able to log which one answered.
func TestReportsTheResolvedModel(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"model":"jev-1.13.0","answers":{"is_urgent":{"type":"noul","noul":0.5}},"usage":{"input_tokens":10,"output_tokens":2}}`)
	})

	req := batch()
	req.Model = "jev-latest"
	got, err := j.Judge(context.Background(), req)
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if got.Model != "jev-1.13.0" {
		t.Errorf("Model = %q, want the resolved jev-1.13.0", got.Model)
	}
}

// Jev serves output tokens free. That zero is a real price, not a missing
// one — TokensReported has to stay true so cost accounting does not mistake
// a free output for an unpriced call.
func TestFreeOutputTokensStillCountAsReported(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"model":"jev-1.13.0","answers":{"is_urgent":{"type":"noul","noul":0.5}},"usage":{"input_tokens":312,"output_tokens":0}}`)
	})

	got, err := j.Judge(context.Background(), batch())
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if !got.Usage.TokensReported {
		t.Error("TokensReported = false, want true — the provider did report counts")
	}
	if got.Usage.InputTokens != 312 || got.Usage.OutputTokens != 0 {
		t.Errorf("Usage = %+v, want 312 in / 0 out", got.Usage)
	}
	if got.Usage.TotalTokens != 312 {
		t.Errorf("TotalTokens = %d, want 312", got.Usage.TotalTokens)
	}
}

func TestErrorStatusMapping(t *testing.T) {
	cases := []struct {
		name      string
		status    int
		body      string
		wantClass llmgate.ErrorClass
		retryable bool
	}{
		{"401 is auth", 401, `{"message":"invalid key"}`, llmgate.ErrClassAuth, false},
		{"403 is auth", 403, `{"message":"forbidden"}`, llmgate.ErrClassAuth, false},
		{"422 is a bad request", 422, `{"detail":[{"loc":["questions","q"],"msg":"field required"}]}`, llmgate.ErrClassBadRequest, false},
		{"429 is rate limited", 429, `{"message":"slow down"}`, llmgate.ErrClassRateLimited, true},
		{"500 is transient", 500, `{"message":"boom"}`, llmgate.ErrClassTransient, true},
		{"529 overloaded is transient", 529, `{"message":"overloaded"}`, llmgate.ErrClassTransient, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				io.WriteString(w, tc.body)
			})

			_, err := j.Judge(context.Background(), batch())
			if err == nil {
				t.Fatal("Judge() = nil error, want a failure")
			}
			if got := llmgate.Classify(err); got != tc.wantClass {
				t.Errorf("Classify = %v, want %v", got, tc.wantClass)
			}
			if got := llmgate.IsTransient(err); got != tc.retryable {
				t.Errorf("IsTransient = %v, want %v", got, tc.retryable)
			}
		})
	}
}

// 422 is the one that matters most. llmgate's generic status mapping leaves
// it Unknown, and the retry middleware retries Unknown by default — so a
// deterministic validation failure would be sent four times to be told the
// same thing.
func TestValidationFailureIsNotRetryable(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(422)
		io.WriteString(w, `{"detail":"questions.q.criteria: at least 2 levels required"}`)
	})

	_, err := j.Judge(context.Background(), batch())
	if err == nil {
		t.Fatal("Judge() = nil error, want a 422 failure")
	}
	if llmgate.IsTransient(err) {
		t.Error("a 422 is retryable, but it is deterministic and must not be")
	}
	// The offending field is the difference between a fixable error and a
	// shrug, so the detail has to survive into the message.
	if !strings.Contains(err.Error(), "criteria") {
		t.Errorf("error = %q, want it to carry the offending field from the 422 detail", err)
	}
}

func TestRetryAfterIsHonoured(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(429)
		io.WriteString(w, `{"message":"slow down"}`)
	})

	_, err := j.Judge(context.Background(), batch())
	if err == nil {
		t.Fatal("Judge() = nil error, want a 429")
	}
	d, ok := llmgate.RetryAfter(err)
	if !ok {
		t.Fatal("RetryAfter reported nothing, want the header's 7s")
	}
	if d != 7*time.Second {
		t.Errorf("RetryAfter = %v, want 7s", d)
	}
}

// A proxy or gateway answering instead of the API returns HTML or a plain
// string with a 200. That is not a model failure and must not look like one.
func TestNonJSONSuccessBodyIsAGatewayError(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `<html><body>502 Bad Gateway</body></html>`)
	})

	_, err := j.Judge(context.Background(), batch())
	if err == nil {
		t.Fatal("Judge() = nil error, want a gateway failure")
	}
	if got := llmgate.Classify(err); got != llmgate.ErrClassGateway {
		t.Errorf("Classify = %v, want gateway", got)
	}
	if llmgate.IsTransient(err) {
		t.Error("a gateway error is retryable, but repeating it reproduces it")
	}
}

// An answer type this package does not know about has to fail loudly. A
// silently dropped answer surfaces much later as ErrAnswerMissing from an
// accessor, a long way from the cause.
func TestUnknownAnswerTypeFails(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, `{"is_urgent":{"type":"quaternion","value":42}}`)
	})

	_, err := j.Judge(context.Background(), batch())
	if err == nil {
		t.Fatal("Judge() = nil error, want an unknown-type failure")
	}
	if !strings.Contains(err.Error(), "quaternion") {
		t.Errorf("error = %q, want it to name the unknown type", err)
	}
}

func TestContextGuardRejectsAnOversizedBatch(t *testing.T) {
	var called bool
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeAnswers(w, `{"q":{"type":"noul","noul":0.5}}`)
	})

	// The stub estimator counts bytes, so a state this long blows the
	// 64k total budget.
	_, err := j.Judge(context.Background(), llmgate.JudgeRequest{
		State:     strings.Repeat("x", typesafe.MaxTotalTokens+1),
		Questions: map[string]llmgate.Question{"q": llmgate.Noul{Instructions: "urgent?"}},
	})

	if err == nil {
		t.Fatal("Judge() = nil error, want a context-length failure")
	}
	if got := llmgate.Classify(err); got != llmgate.ErrClassContextLength {
		t.Errorf("Classify = %v, want context_length", got)
	}
	if called {
		t.Error("the request was sent; the guard exists to catch this before the round-trip")
	}
	if llmgate.IsTransient(err) {
		t.Error("an oversized request is retryable, but retrying cannot help")
	}
}

func TestContextGuardRejectsOneOversizedQuestion(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, `{"big":{"type":"noul","noul":0.5}}`)
	})

	// Under the 64k total, but the single question plus the state breaks
	// the 32k per-question limit.
	_, err := j.Judge(context.Background(), llmgate.JudgeRequest{
		State: "small",
		Questions: map[string]llmgate.Question{
			"big": llmgate.Noul{Instructions: strings.Repeat("y", typesafe.MaxStateAndQuestionTokens+1)},
		},
	})

	if err == nil {
		t.Fatal("Judge() = nil error, want a per-question context failure")
	}
	if got := llmgate.Classify(err); got != llmgate.ErrClassContextLength {
		t.Errorf("Classify = %v, want context_length", got)
	}
	if !strings.Contains(err.Error(), "big") {
		t.Errorf("error = %q, want it to name the offending question", err)
	}
}

func TestSkipContextGuardSendsAnyway(t *testing.T) {
	var called bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeAnswers(w, `{"q":{"type":"noul","noul":0.5}}`)
	}))
	t.Cleanup(srv.Close)

	j, err := typesafe.New(typesafe.Config{
		APIKey:           "k",
		BaseURL:          srv.URL,
		CountTokens:      func(s string) (int, error) { return len(s), nil },
		SkipContextGuard: true,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := j.Judge(context.Background(), llmgate.JudgeRequest{
		State:     strings.Repeat("x", typesafe.MaxTotalTokens+1),
		Questions: map[string]llmgate.Question{"q": llmgate.Noul{Instructions: "urgent?"}},
	}); err != nil {
		t.Fatalf("Judge with the guard off: %v", err)
	}
	if !called {
		t.Error("the guard was skipped but the request was still withheld")
	}
}

// A malformed question is caught before the request is built, so the
// caller does not pay a round-trip to be told what the local types already
// know.
func TestMalformedQuestionNeverReachesTheWire(t *testing.T) {
	var called bool
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		called = true
		writeAnswers(w, `{}`)
	})

	_, err := j.Judge(context.Background(), llmgate.JudgeRequest{
		State:     "state",
		Questions: map[string]llmgate.Question{"bad": llmgate.Score{Instructions: "rate", Levels: []string{"only"}}},
	})

	if !errors.Is(err, llmgate.ErrQuestionInvalid) {
		t.Errorf("Judge() error = %v, want ErrQuestionInvalid", err)
	}
	if called {
		t.Error("a locally-invalid question was sent to the API")
	}
}

func TestNewRequiresAnAPIKey(t *testing.T) {
	t.Setenv(typesafe.EnvAPIKey, "")
	if _, err := typesafe.New(typesafe.Config{}); err == nil {
		t.Error("New() with no key = nil error, want a failure at construction")
	}
}

func TestNewReadsTheKeyFromTheEnvironment(t *testing.T) {
	t.Setenv(typesafe.EnvAPIKey, "env-key")
	if _, err := typesafe.New(typesafe.Config{}); err != nil {
		t.Errorf("New() with the env key set: %v", err)
	}
}

func TestContextCancellationPropagates(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, `{"is_urgent":{"type":"noul","noul":0.5}}`)
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := j.Judge(ctx, batch()); err == nil {
		t.Error("Judge() with a cancelled context = nil error, want a failure")
	}
}

func TestEstimateTokensIsInTheRightNeighbourhood(t *testing.T) {
	// Not asserting an exact count — the estimator is explicitly an
	// approximation for a tokenizer TypeSafe does not publish. The
	// contract is only that it is monotonic and non-zero.
	short, err := typesafe.EstimateTokens("hello world")
	if err != nil {
		t.Fatalf("EstimateTokens: %v", err)
	}
	if short == 0 {
		t.Error("EstimateTokens returned 0 for non-empty text")
	}

	long, err := typesafe.EstimateTokens(strings.Repeat("hello world ", 100))
	if err != nil {
		t.Fatalf("EstimateTokens: %v", err)
	}
	if long <= short {
		t.Errorf("EstimateTokens is not monotonic: %d for long text vs %d for short", long, short)
	}
}

// writeAnswers writes a well-formed success envelope around an answers map.
func writeAnswers(w http.ResponseWriter, answers string) {
	io.WriteString(w, `{"model":"jev-1.13.0","answers":`+answers+`,"usage":{"input_tokens":312,"output_tokens":48}}`)
}
