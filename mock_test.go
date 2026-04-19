package llmgate_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
)

// compile-time check: Mock satisfies Client.
var _ llmgate.Client = (*llmgate.Mock)(nil)

func TestMockReplyCanned(t *testing.T) {
	m := &llmgate.Mock{Reply: "hello there"}
	resp, err := m.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "hello there" {
		t.Fatalf("Content = %q, want %q", resp.Content, "hello there")
	}
	if m.Calls() != 1 {
		t.Fatalf("Calls = %d, want 1", m.Calls())
	}
	prompts := m.LastPrompts()
	if len(prompts) != 1 || prompts[0] != "hi" {
		t.Fatalf("LastPrompts = %v, want [hi]", prompts)
	}
}

func TestMockRespondFn(t *testing.T) {
	m := &llmgate.Mock{
		Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
			return &llmgate.Response{Content: "from fn", Model: req.Model}, nil
		},
	}
	resp, err := m.Complete(context.Background(), llmgate.Request{
		Model:    "claude-test",
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "x"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "from fn" || resp.Model != "claude-test" {
		t.Fatalf("unexpected response %+v", resp)
	}
}

func TestMockCountTokens(t *testing.T) {
	m := &llmgate.Mock{TokensPerCall: 42}
	n, err := m.CountTokens(context.Background(), "anything")
	if err != nil || n != 42 {
		t.Fatalf("CountTokens = (%d, %v), want (42, nil)", n, err)
	}
}

// TestWithRetriesSucceedsAfterTransientError verifies the retry middleware
// re-invokes the inner Client after a transient error and returns success.
func TestWithRetriesSucceedsAfterTransientError(t *testing.T) {
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
	client := llmgate.WithRetries(llmgate.RetryConfig{
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

// TestWithRetriesGivesUp verifies that after MaxRetries+1 attempts the last
// error is surfaced unchanged.
func TestWithRetriesGivesUp(t *testing.T) {
	wantErr := errors.New("still broken")
	var calls int32
	inner := &llmgate.Mock{
		Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
			atomic.AddInt32(&calls, 1)
			return nil, wantErr
		},
	}
	client := llmgate.WithRetries(llmgate.RetryConfig{
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
