package llmgate_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
)

func TestClassify(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want llmgate.ErrorClass
	}{
		{"nil", nil, llmgate.ErrClassUnknown},
		{"rate limit 429", errors.New("HTTP 429: too many requests"), llmgate.ErrClassRateLimited},
		{"rate limit text", errors.New("rate limit exceeded"), llmgate.ErrClassRateLimited},
		{"auth 401", errors.New("HTTP 401 unauthorized"), llmgate.ErrClassAuth},
		{"invalid api key", errors.New("invalid_api_key provided"), llmgate.ErrClassAuth},
		{"bad request", errors.New("HTTP 400 bad request"), llmgate.ErrClassBadRequest},
		{"context length", errors.New("context length exceeded"), llmgate.ErrClassContextLength},
		{"content filter", errors.New("response blocked by safety filter"), llmgate.ErrClassContent},
		{"timeout", errors.New("request timeout"), llmgate.ErrClassTimeout},
		{"500", errors.New("HTTP 500 internal server error"), llmgate.ErrClassTransient},
		{"bad gateway", errors.New("bad gateway"), llmgate.ErrClassTransient},
		{"ctx canceled", context.Canceled, llmgate.ErrClassCanceled},
		{"ctx deadline", context.DeadlineExceeded, llmgate.ErrClassTimeout},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := llmgate.Classify(c.err); got != c.want {
				t.Fatalf("Classify(%v) = %d, want %d", c.err, got, c.want)
			}
		})
	}
}

func TestPredicates(t *testing.T) {
	if !llmgate.IsRateLimited(errors.New("429 rate limit")) {
		t.Fatalf("IsRateLimited should match 429")
	}
	if llmgate.IsAuth(errors.New("429")) {
		t.Fatalf("IsAuth should not match 429")
	}
	if !llmgate.IsAuth(errors.New("401 unauthorized")) {
		t.Fatalf("IsAuth should match 401")
	}
	if !llmgate.IsTransient(errors.New("503 service unavailable")) {
		t.Fatalf("IsTransient should match 503")
	}
}

// TestWithRetriesSkipsAuth verifies auth errors are NOT retried by the
// default retry predicate.
func TestWithRetriesSkipsAuth(t *testing.T) {
	var calls int32
	inner := &llmgate.Mock{
		Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
			atomic.AddInt32(&calls, 1)
			return nil, errors.New("401 unauthorized: invalid_api_key")
		},
	}
	client := llmgate.WithRetries(llmgate.RetryConfig{
		MaxRetries: 5,
		BaseDelay:  1 * time.Millisecond,
		MaxDelay:   2 * time.Millisecond,
	})(inner)

	_, err := client.Complete(context.Background(), llmgate.Request{})
	if err == nil {
		t.Fatalf("expected error")
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("calls = %d, want 1 (auth errors must not retry)", got)
	}
}
