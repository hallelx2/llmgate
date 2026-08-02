package cache_test

import (
	"context"
	"testing"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/middleware/cache"
)

// counting returns a distinct reply per call so a wrong cache hit is
// visible in the content rather than silently plausible.
type counting struct {
	n     int
	tools []llmgate.ToolCall
}

func (c *counting) Complete(context.Context, llmgate.Request) (*llmgate.Response, error) {
	c.n++
	return &llmgate.Response{
		Content:   string(rune('A' + c.n - 1)),
		ToolCalls: c.tools,
	}, nil
}

func (c *counting) CountTokens(context.Context, string) (int, error) { return 0, nil }

// distinct asserts two requests are treated as different cache entries.
func distinct(t *testing.T, name string, a, b llmgate.Request) {
	t.Helper()
	inner := &counting{}
	c := cache.New(cache.Config{})(inner)

	first, err := c.Complete(context.Background(), a)
	if err != nil {
		t.Fatalf("%s: first Complete: %v", name, err)
	}
	second, err := c.Complete(context.Background(), b)
	if err != nil {
		t.Fatalf("%s: second Complete: %v", name, err)
	}

	if second.FromCache {
		t.Errorf("%s: second request served from cache — these must not share a key", name)
	}
	if first.Content == second.Content {
		t.Errorf("%s: both requests returned %q, so the key collided", name, first.Content)
	}
	if inner.n != 2 {
		t.Errorf("%s: inner client called %d times, want 2", name, inner.n)
	}
}

// TestToolChoiceInKey: "required" forces a call and "none" forbids one.
// Identical otherwise, they must not share an answer.
func TestToolChoiceInKey(t *testing.T) {
	base := llmgate.Request{
		Model:    "m",
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
		Tools:    []llmgate.ToolDef{{Name: "search"}},
	}
	a, b := base, base
	a.ToolChoice = "required"
	b.ToolChoice = "none"
	distinct(t, "ToolChoice", a, b)
}

// TestAssistantToolCallsInKey is the tool-loop collision. The assistant
// turn requesting a call usually has empty Content — the calls *are* the
// payload — so hashing only role and content makes two different loop
// states identical.
func TestAssistantToolCallsInKey(t *testing.T) {
	mk := func(arg string) llmgate.Request {
		return llmgate.Request{
			Model: "m",
			Messages: []llmgate.Message{
				{Role: llmgate.RoleUser, Content: "find it"},
				{Role: llmgate.RoleAssistant, Content: "", ToolCalls: []llmgate.ToolCall{
					{ID: "t1", Name: "search", Input: []byte(`{"q":"` + arg + `"}`)},
				}},
				{Role: llmgate.RoleTool, ToolCallID: "t1", Content: "result"},
			},
		}
	}
	distinct(t, "assistant ToolCalls", mk("alpha"), mk("beta"))
}

// TestToolCallIDInKey: the ID links a result to its request, so two
// otherwise identical results answering different calls must differ.
func TestToolCallIDInKey(t *testing.T) {
	mk := func(id string) llmgate.Request {
		return llmgate.Request{
			Model: "m",
			Messages: []llmgate.Message{
				{Role: llmgate.RoleUser, Content: "q"},
				{Role: llmgate.RoleTool, ToolCallID: id, Content: "same result text"},
			},
		}
	}
	distinct(t, "ToolCallID", mk("call_1"), mk("call_2"))
}

// TestSamplingPresenceInKey: an unset temperature means "provider
// default" (~1.0) and an explicit 0 means deterministic. Very different
// answers, so they cannot share a key.
func TestSamplingPresenceInKey(t *testing.T) {
	base := llmgate.Request{
		Model:    "m",
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
	}
	a, b := base, base
	b.Temperature = llmgate.Float64(0)
	distinct(t, "Temperature presence", a, b)
}

// TestCachedResponseIsDeepCopied: a caller that rewrites tool-call
// arguments in place must not corrupt the entry for every later hit.
func TestCachedResponseIsDeepCopied(t *testing.T) {
	inner := &counting{tools: []llmgate.ToolCall{
		{ID: "t1", Name: "search", Input: []byte(`{"q":"original"}`)},
	}}
	c := cache.New(cache.Config{})(inner)

	req := llmgate.Request{
		Model:    "m",
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
	}

	if _, err := c.Complete(context.Background(), req); err != nil {
		t.Fatalf("priming Complete: %v", err)
	}

	hit, err := c.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("cached Complete: %v", err)
	}
	if !hit.FromCache {
		t.Fatal("second identical request should have hit the cache")
	}

	// Mutate the returned arguments in place, as a caller unmarshalling
	// and rewriting them would.
	copy(hit.ToolCalls[0].Input, []byte(`{"q":"MUTATED!"}`))

	again, err := c.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("third Complete: %v", err)
	}
	if got := string(again.ToolCalls[0].Input); got != `{"q":"original"}` {
		t.Errorf("cached tool input = %s, want the original — the entry was aliased", got)
	}
}
