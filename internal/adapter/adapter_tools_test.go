package adapter

import (
	"context"
	"testing"

	"github.com/tmc/langchaingo/llms"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/pricing"
)

// fakeModel is a langchaingo llms.Model that records what it was called
// with (after applying the functional options) and returns a canned
// ContentResponse. It lets us assert the adapter forwards tools and maps
// tool calls back without touching a real provider.
type fakeModel struct {
	gotOpts  llms.CallOptions
	gotMsgs  []llms.MessageContent
	response *llms.ContentResponse
}

func (f *fakeModel) GenerateContent(_ context.Context, msgs []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	f.gotMsgs = msgs
	for _, o := range options {
		o(&f.gotOpts)
	}
	return f.response, nil
}

func (f *fakeModel) Call(_ context.Context, _ string, _ ...llms.CallOption) (string, error) {
	return "", nil
}

func newFake(resp *llms.ContentResponse) *fakeModel { return &fakeModel{response: resp} }

func textResponse(content string) *llms.ContentResponse {
	return &llms.ContentResponse{Choices: []*llms.ContentChoice{{
		Content:        content,
		StopReason:     "stop",
		GenerationInfo: map[string]any{"PromptTokens": 100, "CompletionTokens": 20},
	}}}
}

// TestToolsForwarded: a Request with Tools and a ToolChoice reaches the
// underlying model as llms.WithTools / llms.WithToolChoice, with the JSON
// schema unmarshalled into the function parameters.
func TestToolsForwarded(t *testing.T) {
	fm := newFake(textResponse("ok"))
	a := NewAdapter(fm, llmgate.ProviderOpenAI, "gpt-4o-mini", true)

	_, err := a.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
		Tools: []llmgate.ToolDef{{
			Name:        "get_pages",
			Description: "fetch pages",
			InputSchema: []byte(`{"type":"object","properties":{"start":{"type":"integer"}}}`),
		}},
		ToolChoice: "auto",
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if len(fm.gotOpts.Tools) != 1 {
		t.Fatalf("expected 1 tool forwarded, got %d", len(fm.gotOpts.Tools))
	}
	tool := fm.gotOpts.Tools[0]
	if tool.Type != "function" || tool.Function == nil || tool.Function.Name != "get_pages" {
		t.Fatalf("tool not forwarded correctly: %+v", tool)
	}
	if tool.Function.Parameters == nil {
		t.Fatalf("expected schema unmarshalled into Parameters, got nil")
	}
	if fm.gotOpts.ToolChoice != "auto" {
		t.Fatalf("ToolChoice = %v, want auto", fm.gotOpts.ToolChoice)
	}
}

// TestToolChoiceNamedTool: a non-keyword ToolChoice forces a specific tool.
func TestToolChoiceNamedTool(t *testing.T) {
	fm := newFake(textResponse("ok"))
	a := NewAdapter(fm, llmgate.ProviderOpenAI, "gpt-4o-mini", true)

	_, err := a.Complete(context.Background(), llmgate.Request{
		Messages:   []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
		Tools:      []llmgate.ToolDef{{Name: "done"}},
		ToolChoice: "done",
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	tc, ok := fm.gotOpts.ToolChoice.(llms.ToolChoice)
	if !ok {
		t.Fatalf("ToolChoice type = %T, want llms.ToolChoice", fm.gotOpts.ToolChoice)
	}
	if tc.Function == nil || tc.Function.Name != "done" {
		t.Fatalf("forced tool choice wrong: %+v", tc)
	}
}

// TestToolCallsMappedBack: tool calls in the provider response surface on
// the llmgate Response with name + raw JSON arguments preserved.
func TestToolCallsMappedBack(t *testing.T) {
	resp := &llms.ContentResponse{Choices: []*llms.ContentChoice{{
		StopReason: "tool_calls",
		ToolCalls: []llms.ToolCall{{
			ID:           "call_1",
			Type:         "function",
			FunctionCall: &llms.FunctionCall{Name: "get_pages", Arguments: `{"start":5,"end":7}`},
		}},
		GenerationInfo: map[string]any{"PromptTokens": 50, "CompletionTokens": 10},
	}}}
	fm := newFake(resp)
	a := NewAdapter(fm, llmgate.ProviderOpenAI, "gpt-4o-mini", true)

	out, err := a.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
		Tools:    []llmgate.ToolDef{{Name: "get_pages"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(out.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(out.ToolCalls))
	}
	call := out.ToolCalls[0]
	if call.ID != "call_1" || call.Name != "get_pages" || string(call.Input) != `{"start":5,"end":7}` {
		t.Fatalf("tool call mapped wrong: %+v (input=%s)", call, call.Input)
	}
}

// TestFuncCallFallback: providers that populate the singular FuncCall (not
// ToolCalls) still surface as a single ToolCall.
func TestFuncCallFallback(t *testing.T) {
	resp := &llms.ContentResponse{Choices: []*llms.ContentChoice{{
		FuncCall:       &llms.FunctionCall{Name: "done", Arguments: `{"answer":"42"}`},
		GenerationInfo: map[string]any{},
	}}}
	a := NewAdapter(newFake(resp), llmgate.ProviderGemini, "gemini-2.5-flash", true)

	out, err := a.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(out.ToolCalls) != 1 || out.ToolCalls[0].Name != "done" {
		t.Fatalf("FuncCall fallback failed: %+v", out.ToolCalls)
	}
}

// TestPricedFlag: a known model marks Usage.Priced true with a non-zero
// cost; an unknown model marks it false with zero cost (not "free").
func TestPricedFlag(t *testing.T) {
	known := NewAdapter(newFake(textResponse("ok")), llmgate.ProviderOpenAI, "gpt-4o-mini", true)
	out, err := known.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if !out.Usage.Priced || out.Usage.CostUSD <= 0 {
		t.Fatalf("known model should be priced with cost>0, got priced=%v cost=%v", out.Usage.Priced, out.Usage.CostUSD)
	}

	unknown := NewAdapter(newFake(textResponse("ok")), llmgate.ProviderOpenAI, "totally-unknown-model", true)
	out2, err := unknown.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if out2.Usage.Priced || out2.Usage.CostUSD != 0 {
		t.Fatalf("unknown model should be unpriced with 0 cost, got priced=%v cost=%v", out2.Usage.Priced, out2.Usage.CostUSD)
	}
}

// TestGLMPriced guards the specific regression that made the benchmark read
// $0: glm-4.6 must now compute a real cost.
func TestGLMPriced(t *testing.T) {
	if cost, ok := pricing.ComputeWithOK("glm-4.6", 10000, 1000); !ok || cost <= 0 {
		t.Fatalf("glm-4.6 should be priced with cost>0, got ok=%v cost=%v", ok, cost)
	}
}

// TestMessageConversionToolLoop: an assistant tool-call turn and a
// tool-result turn convert into the right langchaingo parts and roles.
func TestMessageConversionToolLoop(t *testing.T) {
	msgs := []llmgate.Message{
		{Role: llmgate.RoleUser, Content: "where is the debt note?"},
		{Role: llmgate.RoleAssistant, ToolCalls: []llmgate.ToolCall{{
			ID: "c1", Name: "get_pages", Input: []byte(`{"start":5}`),
		}}},
		{Role: llmgate.RoleTool, ToolCallID: "c1", Content: "page 5 text..."},
	}
	out := toLangchainMessages(msgs, false, nil)
	if len(out) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(out))
	}

	// Assistant turn: pure tool call, no empty text placeholder.
	if out[1].Role != llms.ChatMessageTypeAI {
		t.Fatalf("assistant role = %v", out[1].Role)
	}
	if len(out[1].Parts) != 1 {
		t.Fatalf("assistant should have exactly 1 part (the tool call), got %d", len(out[1].Parts))
	}
	tcPart, ok := out[1].Parts[0].(llms.ToolCall)
	if !ok || tcPart.FunctionCall == nil || tcPart.FunctionCall.Name != "get_pages" {
		t.Fatalf("assistant tool call part wrong: %+v", out[1].Parts[0])
	}

	// Tool turn: role tool, ToolCallResponse carrying the result.
	if out[2].Role != llms.ChatMessageTypeTool {
		t.Fatalf("tool role = %v", out[2].Role)
	}
	resp, ok := out[2].Parts[0].(llms.ToolCallResponse)
	if !ok || resp.ToolCallID != "c1" || resp.Content != "page 5 text..." {
		t.Fatalf("tool response part wrong: %+v", out[2].Parts[0])
	}
}
