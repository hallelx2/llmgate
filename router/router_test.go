package router_test

import (
	"context"
	"errors"
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

func TestSurfacesLastError(t *testing.T) {
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
	if got := err.Error(); got != "503 from b" {
		t.Fatalf("err = %q, want last error %q", got, "503 from b")
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
