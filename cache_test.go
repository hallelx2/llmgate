package llmgate_test

import (
	"context"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
)

func TestCacheHit(t *testing.T) {
	inner := &llmgate.Mock{Reply: "cached"}
	client := llmgate.WithCache(llmgate.CacheConfig{Capacity: 16})(inner)

	req := llmgate.Request{
		Model:    "gpt-4o-mini",
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hello"}},
	}

	r1, err := client.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("first: %v", err)
	}
	if r1.FromCache {
		t.Fatalf("first call should not be FromCache")
	}

	r2, err := client.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("second: %v", err)
	}
	if !r2.FromCache {
		t.Fatalf("second call should be FromCache")
	}
	if r2.Content != "cached" {
		t.Fatalf("content = %q", r2.Content)
	}
	if inner.Calls() != 1 {
		t.Fatalf("inner called %d times, want 1", inner.Calls())
	}
	if r2.Usage.CostUSD != 0 {
		t.Fatalf("cached Usage.CostUSD should be 0, got %v", r2.Usage.CostUSD)
	}
}

func TestCacheMissOnDifferentMaxTokens(t *testing.T) {
	inner := &llmgate.Mock{Reply: "ok"}
	client := llmgate.WithCache(llmgate.CacheConfig{})(inner)

	base := llmgate.Request{
		Model:    "m",
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "x"}},
	}
	a := base
	a.MaxTokens = 100
	b := base
	b.MaxTokens = 200

	if _, err := client.Complete(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	if _, err := client.Complete(context.Background(), b); err != nil {
		t.Fatal(err)
	}
	if inner.Calls() != 2 {
		t.Fatalf("inner calls = %d, want 2 (distinct MaxTokens)", inner.Calls())
	}
}

func TestCacheTTLExpiry(t *testing.T) {
	now := time.Date(2026, 4, 19, 12, 0, 0, 0, time.UTC)
	clock := &now
	inner := &llmgate.Mock{Reply: "ok"}
	client := llmgate.WithCache(llmgate.CacheConfig{
		TTL: 1 * time.Minute,
		Now: func() time.Time { return *clock },
	})(inner)

	req := llmgate.Request{Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "x"}}}
	if _, err := client.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	// Advance past TTL.
	*clock = now.Add(2 * time.Minute)
	r2, err := client.Complete(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if r2.FromCache {
		t.Fatalf("expected cache miss after TTL expiry")
	}
	if inner.Calls() != 2 {
		t.Fatalf("inner calls = %d, want 2", inner.Calls())
	}
}

func TestCacheCapacityEviction(t *testing.T) {
	inner := &llmgate.Mock{Reply: "ok"}
	client := llmgate.WithCache(llmgate.CacheConfig{Capacity: 2})(inner)
	mkReq := func(s string) llmgate.Request {
		return llmgate.Request{Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: s}}}
	}
	client.Complete(context.Background(), mkReq("a"))
	client.Complete(context.Background(), mkReq("b"))
	client.Complete(context.Background(), mkReq("c")) // evicts "a"
	// "a" should miss now.
	r, err := client.Complete(context.Background(), mkReq("a"))
	if err != nil {
		t.Fatal(err)
	}
	if r.FromCache {
		t.Fatalf("expected eviction to cause miss on a")
	}
}
