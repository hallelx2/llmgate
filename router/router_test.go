package router_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/router"
)

func TestFallsOverOnRateLimit(t *testing.T) {
	primary := &llmgate.Mock{
		Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
			return nil, errors.New("rate limit exceeded")
		},
	}
	secondary := &llmgate.Mock{Reply: "from secondary"}

	r, err := router.New(router.Config{
		Clients: []llmgate.Client{primary, secondary},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "from secondary" {
		t.Fatalf("Content = %q, want from secondary", resp.Content)
	}
	if primary.Calls() != 1 || secondary.Calls() != 1 {
		t.Fatalf("primary=%d secondary=%d, want 1,1", primary.Calls(), secondary.Calls())
	}
}

func TestSurfacesAllErrors(t *testing.T) {
	a := &llmgate.Mock{Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
		return nil, errors.New("429 from a")
	}}
	b := &llmgate.Mock{Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
		return nil, errors.New("503 from b")
	}}
	r, err := router.New(router.Config{Clients: []llmgate.Client{a, b}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = r.Complete(context.Background(), llmgate.Request{})
	if err == nil {
		t.Fatalf("expected error")
	}

	// The error should be a *RouteError containing both failures.
	var routeErr *router.RouteError
	if !errors.As(err, &routeErr) {
		t.Fatalf("expected *RouteError, got %T: %v", err, err)
	}

	// Primary error is from provider A (429).
	if primary := routeErr.Primary(); primary == nil || !strings.Contains(primary.Error(), "429 from a") {
		t.Fatalf("Primary() = %v, want '429 from a'", primary)
	}

	// Last error is from provider B (503).
	if last := routeErr.Last(); last == nil || !strings.Contains(last.Error(), "503 from b") {
		t.Fatalf("Last() = %v, want '503 from b'", last)
	}

	// Error message mentions both.
	msg := err.Error()
	if !strings.Contains(msg, "429 from a") || !strings.Contains(msg, "503 from b") {
		t.Fatalf("Error() = %q, want both errors mentioned", msg)
	}

	// errors.Is should unwrap to primary for Classify to work.
	if !errors.Is(err, routeErr.Primary()) {
		t.Fatalf("errors.Is should match primary error")
	}
}

func TestDoesNotFallOverOnAuth(t *testing.T) {
	a := &llmgate.Mock{Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
		return nil, errors.New("401 unauthorized")
	}}
	b := &llmgate.Mock{Reply: "should not be reached"}
	r, err := router.New(router.Config{Clients: []llmgate.Client{a, b}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = r.Complete(context.Background(), llmgate.Request{})
	if err == nil {
		t.Fatalf("expected auth error")
	}
	if b.Calls() != 0 {
		t.Fatalf("secondary called %d times; should have been skipped", b.Calls())
	}
}

func TestRequiresClients(t *testing.T) {
	if _, err := router.New(router.Config{}); err == nil {
		t.Fatalf("expected error for empty Clients")
	}
}

func TestOnRateLimitOnlyPolicy(t *testing.T) {
	a := &llmgate.Mock{Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
		return nil, errors.New("503 transient")
	}}
	b := &llmgate.Mock{Reply: "b"}
	r, err := router.New(router.Config{
		Clients:  []llmgate.Client{a, b},
		Fallback: router.OnRateLimit,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = r.Complete(context.Background(), llmgate.Request{})
	if err == nil {
		t.Fatalf("expected 503 to surface under OnRateLimit")
	}
	if b.Calls() != 0 {
		t.Fatalf("secondary should not be called under OnRateLimit")
	}
}

func TestMockLastMessages(t *testing.T) {
	m := &llmgate.Mock{Reply: "ok"}
	m.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{
			{Role: llmgate.RoleSystem, Content: "you are helpful"},
			{Role: llmgate.RoleUser, Content: "hello"},
		},
	})
	msgs := m.LastMessages()
	if len(msgs) != 1 {
		t.Fatalf("LastMessages len = %d, want 1", len(msgs))
	}
	if len(msgs[0]) != 2 {
		t.Fatalf("msgs[0] len = %d, want 2 (system + user)", len(msgs[0]))
	}
	if msgs[0][0].Role != llmgate.RoleSystem || msgs[0][0].Content != "you are helpful" {
		t.Fatalf("msgs[0][0] = %v, want system/you are helpful", msgs[0][0])
	}
}

func TestLLMErrorClassification(t *testing.T) {
	err := llmgate.NewLLMError(llmgate.ProviderAnthropic, 429, "rate limited", nil)
	if llmgate.Classify(err) != llmgate.ErrClassRateLimited {
		t.Fatalf("Classify(LLMError 429) = %v, want RateLimited", llmgate.Classify(err))
	}

	err2 := llmgate.NewLLMError(llmgate.ProviderOpenAI, 401, "bad key", nil)
	if llmgate.Classify(err2) != llmgate.ErrClassAuth {
		t.Fatalf("Classify(LLMError 401) = %v, want Auth", llmgate.Classify(err2))
	}

	// Wrapped LLMError should still be classified.
	wrapped := errors.Join(errors.New("context"), err)
	if llmgate.Classify(wrapped) != llmgate.ErrClassRateLimited {
		t.Fatalf("Classify(wrapped LLMError 429) = %v, want RateLimited", llmgate.Classify(wrapped))
	}
}
