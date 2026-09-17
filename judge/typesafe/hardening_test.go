package typesafe_test

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/judge/typesafe"
)

// Every request carries Authorization: Bearer <key>. A plain-http base URL
// would put a live credential on the wire in cleartext, which is worth
// failing at construction rather than trusting a caller to notice.
func TestPlainHTTPBaseURLIsRefused(t *testing.T) {
	_, err := typesafe.New(typesafe.Config{
		APIKey:  "k",
		BaseURL: "http://api.typesafe.ai",
	})
	if err == nil {
		t.Fatal("New() with an http base URL = nil error, want a refusal")
	}
	if !strings.Contains(err.Error(), "cleartext") {
		t.Errorf("error = %q, want it to explain the credential exposure", err)
	}
}

// Loopback stays allowed — a test server and a local proxy are both http,
// and neither leaves the machine. Without this the package could not be
// tested against httptest at all.
func TestLoopbackHTTPBaseURLIsAllowed(t *testing.T) {
	for _, base := range []string{
		"http://localhost:8080",
		"http://127.0.0.1:8080",
		"http://[::1]:8080",
	} {
		if _, err := typesafe.New(typesafe.Config{APIKey: "k", BaseURL: base}); err != nil {
			t.Errorf("New() with %q = %v, want nil", base, err)
		}
	}
}

func TestNonHTTPBaseURLIsRefused(t *testing.T) {
	_, err := typesafe.New(typesafe.Config{APIKey: "k", BaseURL: "ftp://example.com"})
	if err == nil {
		t.Fatal("New() with an ftp base URL = nil error, want a refusal")
	}
}

// An answer under an ID that was never asked means the response does not
// correspond to the request.
func TestUnrequestedAnswerIsRejected(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, `{"is_urgent":{"type":"noul","noul":0.5},"surprise":{"type":"noul","noul":0.1}}`)
	})

	_, err := j.Judge(context.Background(), batch())
	if err == nil {
		t.Fatal("Judge() = nil error, want a mismatched-response failure")
	}
	if !strings.Contains(err.Error(), "surprise") {
		t.Errorf("error = %q, want it to name the unrequested answer", err)
	}
}

// A Noul answered with a Choice is not a usable answer under a different
// name. Caught at the boundary the caller sees why; left alone it surfaces
// as ErrAnswerType from an accessor much later.
func TestPrimitiveMismatchIsRejected(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, `{"is_urgent":{"type":"choice","choice":"a","probabilities":{"a":1},"confidence":0.9}}`)
	})

	_, err := j.Judge(context.Background(), batch())
	if err == nil {
		t.Fatal("Judge() = nil error, want a primitive-mismatch failure")
	}
	if !strings.Contains(err.Error(), "noul") || !strings.Contains(err.Error(), "choice") {
		t.Errorf("error = %q, want it to name both the asked and answered primitives", err)
	}
}

// An unbounded read turns a broken or hostile endpoint into an OOM here.
// The body is capped, and overflowing the cap is a gateway error.
func TestOversizedResponseBodyIsRejected(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Stream well past the cap without allocating it here.
		chunk := strings.Repeat("x", 1<<20)
		for i := 0; i < 9; i++ {
			if _, err := w.Write([]byte(chunk)); err != nil {
				return
			}
		}
	})

	_, err := j.Judge(context.Background(), batch())
	if err == nil {
		t.Fatal("Judge() = nil error, want an oversized-body failure")
	}
	if got := llmgate.Classify(err); got != llmgate.ErrClassGateway {
		t.Errorf("Classify = %v, want gateway", got)
	}
}

// Validate accepts a non-nil pointer question, so every consumer has to
// handle one. The mock previously called it an unknown kind, which would
// fail a request the real transport sends happily.
func TestMockJudgeHandlesPointerQuestions(t *testing.T) {
	m := &llmgate.MockJudge{}

	got, err := m.Judge(context.Background(), llmgate.JudgeRequest{
		State: "s",
		Questions: map[string]llmgate.Question{
			"n": &llmgate.Noul{Instructions: "urgent?"},
			"c": &llmgate.Choice{Instructions: "which?", Options: llmgate.ChoiceOptions{
				{Name: "a"}, {Name: "b"},
			}},
			"s": &llmgate.Score{Instructions: "how much?", Levels: []string{"low", "high"}},
		},
	})
	if err != nil {
		t.Fatalf("Judge with pointer questions: %v", err)
	}
	if len(got.Answers) != 3 {
		t.Errorf("got %d answers, want 3", len(got.Answers))
	}
	if _, err := got.Noul("n"); err != nil {
		t.Errorf("Noul: %v", err)
	}
	if _, err := got.Choice("c"); err != nil {
		t.Errorf("Choice: %v", err)
	}
	if _, err := got.Score("s"); err != nil {
		t.Errorf("Score: %v", err)
	}
}
