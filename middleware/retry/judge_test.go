package retry_test

import (
	"context"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/middleware/retry"
)

// failingJudge fails with err until it has been called failFirst times.
type failingJudge struct {
	err       error
	failFirst int
	calls     int
}

func (f *failingJudge) Judge(context.Context, llmgate.JudgeRequest) (*llmgate.Judgment, error) {
	f.calls++
	if f.calls <= f.failFirst {
		return nil, f.err
	}
	return &llmgate.Judgment{Model: "jev-1.13.0"}, nil
}

func judgeReq() llmgate.JudgeRequest {
	return llmgate.JudgeRequest{
		State:     "state",
		Questions: map[string]llmgate.Question{"q": llmgate.Noul{Instructions: "urgent?"}},
	}
}

func TestJudgeRetriesTransientFailures(t *testing.T) {
	inner := &failingJudge{err: errTransient, failFirst: 2}
	j := retry.NewJudge(retry.Config{MaxRetries: 3, BaseDelay: time.Nanosecond, MaxDelay: time.Millisecond})(inner)

	got, err := j.Judge(context.Background(), judgeReq())
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if got.Model != "jev-1.13.0" {
		t.Errorf("Model = %q, want the eventual success", got.Model)
	}
	if inner.calls != 3 {
		t.Errorf("inner called %d times, want 3 (2 failures then a success)", inner.calls)
	}
}

// A 422 is deterministic. Retrying it burns the budget four times over to
// be told the same thing, and on a batched call that is four times the
// state as well.
func TestJudgeDoesNotRetryDeterministicFailures(t *testing.T) {
	badRequest := &llmgate.LLMError{
		Class:      llmgate.ErrClassBadRequest,
		StatusCode: 422,
		Provider:   llmgate.ProviderTypeSafe,
		Message:    "questions.q.criteria: at least 2 levels required",
	}
	inner := &failingJudge{err: badRequest, failFirst: 100}
	j := retry.NewJudge(retry.Config{MaxRetries: 5, BaseDelay: time.Nanosecond, MaxDelay: time.Millisecond})(inner)

	if _, err := j.Judge(context.Background(), judgeReq()); err == nil {
		t.Fatal("Judge() = nil error, want the 422")
	}
	if inner.calls != 1 {
		t.Errorf("inner called %d times, want exactly 1 — a 422 must not be retried", inner.calls)
	}
}

func TestJudgeDoesNotRetryAuthFailures(t *testing.T) {
	inner := &failingJudge{
		err:       &llmgate.LLMError{Class: llmgate.ErrClassAuth, StatusCode: 401, Provider: llmgate.ProviderTypeSafe},
		failFirst: 100,
	}
	j := retry.NewJudge(retry.Config{MaxRetries: 5, BaseDelay: time.Nanosecond, MaxDelay: time.Millisecond})(inner)

	if _, err := j.Judge(context.Background(), judgeReq()); err == nil {
		t.Fatal("Judge() = nil error, want the 401")
	}
	if inner.calls != 1 {
		t.Errorf("inner called %d times, want 1 — a bad key does not fix itself", inner.calls)
	}
}

// The provider knows when capacity frees up; backoff only guesses. A 60ms
// Retry-After against a 1ns base means the call cannot return sooner.
func TestJudgeHonoursRetryAfter(t *testing.T) {
	inner := &failingJudge{
		err: &llmgate.LLMError{
			Class:         llmgate.ErrClassRateLimited,
			StatusCode:    429,
			Provider:      llmgate.ProviderTypeSafe,
			RetryAfterDur: 60 * time.Millisecond,
		},
		failFirst: 1,
	}
	j := retry.NewJudge(retry.Config{MaxRetries: 2, BaseDelay: time.Nanosecond, MaxDelay: time.Second})(inner)

	start := time.Now()
	if _, err := j.Judge(context.Background(), judgeReq()); err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if elapsed := time.Since(start); elapsed < 60*time.Millisecond {
		t.Errorf("returned after %v, want at least the 60ms the provider asked for", elapsed)
	}
}

func TestJudgeStopsOnContextCancellation(t *testing.T) {
	inner := &failingJudge{err: errTransient, failFirst: 100}
	j := retry.NewJudge(retry.Config{MaxRetries: 10, BaseDelay: 50 * time.Millisecond, MaxDelay: time.Second})(inner)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := j.Judge(ctx, judgeReq()); err == nil {
		t.Fatal("Judge() = nil error, want the cancellation")
	}
	if inner.calls > 1 {
		t.Errorf("inner called %d times after cancellation, want at most 1", inner.calls)
	}
}

// The same extreme configs that used to panic the Client middleware, on the
// judge path. They share the backoff helpers precisely so this cannot
// regress on only one seam.
func TestJudgeNoPanicOnExtremeConfig(t *testing.T) {
	tests := []struct {
		name       string
		maxRetries int
		base, max  time.Duration
	}{
		{"overflowing attempt count", 40, 500 * time.Millisecond, time.Microsecond},
		{"sub-nanosecond base", 5, 1, time.Microsecond},
		{"zero-ish jitter window", 3, 2, 3},
		{"huge base", 4, time.Duration(1) << 61, time.Millisecond},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			j := retry.NewJudge(retry.Config{
				MaxRetries: tc.maxRetries,
				BaseDelay:  tc.base,
				MaxDelay:   tc.max,
			})(&failingJudge{err: errTransient, failFirst: 1000})

			if _, err := j.Judge(context.Background(), judgeReq()); err == nil {
				t.Fatal("expected the call to fail after exhausting retries")
			}
		})
	}
}
