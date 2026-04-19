package llmgate_test

import (
	"context"
	"testing"

	"github.com/hallelx2/llmgate"
)

// compile-time check: Mock, Anthropic, OpenAI, Gemini all satisfy Client.
var (
	_ llmgate.Client = (*llmgate.Mock)(nil)
	_ llmgate.Client = (*llmgate.Anthropic)(nil)
	_ llmgate.Client = (*llmgate.OpenAI)(nil)
	_ llmgate.Client = (*llmgate.Gemini)(nil)
)

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

func TestStubsReturnNotImplemented(t *testing.T) {
	ctx := context.Background()
	for _, c := range []llmgate.Client{
		llmgate.NewOpenAI(llmgate.OpenAIConfig{}),
		llmgate.NewGemini(llmgate.GeminiConfig{}),
	} {
		_, err := c.Complete(ctx, llmgate.Request{})
		if err != llmgate.ErrNotImplemented {
			t.Fatalf("stub returned %v, want ErrNotImplemented", err)
		}
	}
}
