package retry_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/middleware/retry"
)

// TestSucceedsAfterTransientError verifies the retry middleware
// re-invokes the inner Client after a transient error and returns success.
func TestSucceedsAfterTransientError(t *testing.T) {
	var calls int32
	inner := &llmgate.Mock{
		Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
			n := atomic.AddInt32(&calls, 1)
			if n < 3 {
				return nil, errors.New("transient")
			}
			return &llmgate.Response{Content: "ok"}, nil
		},
	}
	client := retry.New(retry.Config{
		MaxRetries: 5,
		BaseDelay:  1 * time.Millisecond,
		MaxDelay:   2 * time.Millisecond,
	})(inner)

	resp, err := client.Complete(context.Background(), llmgate.Request{})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "ok" {
		t.Fatalf("Content = %q, want %q", resp.Content, "ok")
	}
	if got := atomic.LoadInt32(&calls); got != 3 {
		t.Fatalf("calls = %d, want 3", got)
	}
}

// TestGivesUp verifies that after MaxRetries+1 attempts the last
// error is surfaced unchanged.
func TestGivesUp(t *testing.T) {
	wantErr := errors.New("still broken")
	var calls int32
	inner := &llmgate.Mock{
		Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
			atomic.AddInt32(&calls, 1)
			return nil, wantErr
		},
	}
	client := retry.New(retry.Config{
		MaxRetries: 2,
		BaseDelay:  1 * time.Millisecond,
		MaxDelay:   2 * time.Millisecond,
	})(inner)

	_, err := client.Complete(context.Background(), llmgate.Request{})
	if !errors.Is(err, wantErr) {
		t.Fatalf("err = %v, want %v", err, wantErr)
	}
	if got := atomic.LoadInt32(&calls); got != 3 {
		t.Fatalf("calls = %d, want 3 (initial + 2 retries)", got)
	}
}

// TestSkipsAuth verifies auth errors are NOT retried by the default
// retry predicate.
func TestSkipsAuth(t *testing.T) {
	var calls int32
	inner := &llmgate.Mock{
		Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
			atomic.AddInt32(&calls, 1)
			return nil, errors.New("401 unauthorized: invalid_api_key")
		},
	}
	client := retry.New(retry.Config{
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
