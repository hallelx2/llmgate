package adapter

import (
	"context"
	"testing"

	"github.com/tmc/langchaingo/llms"

	"github.com/hallelx2/llmgate"
)

// unsetTemp is a sentinel written into the fake's CallOptions before the
// adapter applies its options. llms.CallOptions zero-values Temperature to
// 0, so without a sentinel "never set" and "explicitly set to 0" are
// indistinguishable — which is the exact bug under test.
const unsetTemp = -999.0

// completeWith runs one Complete against a canned single-text response and
// returns the CallOptions the adapter actually built.
func completeWith(t *testing.T, req llmgate.Request) llms.CallOptions {
	t.Helper()
	fm := newFake(textResponse("ok"))
	fm.gotOpts.Temperature = unsetTemp
	fm.gotOpts.TopP = unsetTemp

	a := NewAdapter(fm, llmgate.ProviderOpenAI, "gpt-4o-mini", true)
	if len(req.Messages) == 0 {
		req.Messages = []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}}
	}
	if _, err := a.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	return fm.gotOpts
}

// TestTemperatureZeroReachesProvider is the regression test for HAL-524:
// every vectorless call asks for temperature 0 and the old `!= 0` guard
// dropped all of them, silently running the whole engine at the provider
// default.
func TestTemperatureZeroReachesProvider(t *testing.T) {
	got := completeWith(t, llmgate.Request{Temperature: llmgate.Float64(0)})
	if got.Temperature != 0 {
		t.Fatalf("Temperature = %v, want 0 — an explicit zero must reach the provider", got.Temperature)
	}
}

func TestTemperatureUnsetIsNotForwarded(t *testing.T) {
	got := completeWith(t, llmgate.Request{})
	if got.Temperature != unsetTemp {
		t.Fatalf("Temperature = %v, want untouched (%v) — nil must leave the provider default alone", got.Temperature, unsetTemp)
	}
}

func TestTemperatureNonZeroForwarded(t *testing.T) {
	got := completeWith(t, llmgate.Request{Temperature: llmgate.Float64(0.7)})
	if got.Temperature != 0.7 {
		t.Fatalf("Temperature = %v, want 0.7", got.Temperature)
	}
}

func TestTopPAndSeedOptional(t *testing.T) {
	unset := completeWith(t, llmgate.Request{})
	if unset.TopP != unsetTemp {
		t.Fatalf("TopP = %v, want untouched", unset.TopP)
	}
	if unset.Seed != 0 {
		t.Fatalf("Seed = %v, want 0 when unset", unset.Seed)
	}

	set := completeWith(t, llmgate.Request{TopP: llmgate.Float64(0), Seed: llmgate.Int(42)})
	if set.TopP != 0 {
		t.Fatalf("TopP = %v, want explicit 0", set.TopP)
	}
	if set.Seed != 42 {
		t.Fatalf("Seed = %v, want 42", set.Seed)
	}
}

// --- choice folding (HAL-525) -------------------------------------------

// anthropicUsage mirrors the GenerationInfo langchaingo's Anthropic adapter
// stamps onto *every* content block of a single response.
func anthropicUsage() map[string]any {
	return map[string]any{
		"InputTokens":  1000,
		"OutputTokens": 200,
	}
}

func textBlock(s string) *llms.ContentChoice {
	return &llms.ContentChoice{
		Content:        s,
		StopReason:     "end_turn",
		GenerationInfo: anthropicUsage(),
	}
}

func thinkingBlock(thought string) *llms.ContentChoice {
	gi := anthropicUsage()
	gi["ThinkingContent"] = thought
	return &llms.ContentChoice{
		Content:        "", // langchaingo: "Thinking content is not included in output"
		StopReason:     "end_turn",
		GenerationInfo: gi,
	}
}

func toolBlock(id, name, args string) *llms.ContentChoice {
	return &llms.ContentChoice{
		ToolCalls: []llms.ToolCall{{
			ID:           id,
			Type:         "function",
			FunctionCall: &llms.FunctionCall{Name: name, Arguments: args},
		}},
		StopReason:     "tool_use",
		GenerationInfo: anthropicUsage(),
	}
}

// TestFoldChoices covers the shapes langchaingo's Anthropic adapter
// produces. It builds one ContentChoice per *content block*, so reading
// Choices[0] returned an empty answer whenever the model thought first and
// dropped tool calls whenever it narrated before calling.
func TestFoldChoices(t *testing.T) {
	tests := []struct {
		name          string
		choices       []*llms.ContentChoice
		wantContent   string
		wantReasoning string
		wantToolCalls []string // tool names, in order
	}{
		{
			name:        "text only",
			choices:     []*llms.ContentChoice{textBlock("the answer")},
			wantContent: "the answer",
		},
		{
			name:          "thinking then text",
			choices:       []*llms.ContentChoice{thinkingBlock("let me consider"), textBlock("the answer")},
			wantContent:   "the answer",
			wantReasoning: "let me consider",
		},
		{
			name:          "text then tool use",
			choices:       []*llms.ContentChoice{textBlock("looking that up"), toolBlock("t1", "get_pages", `{"start":1}`)},
			wantContent:   "looking that up",
			wantToolCalls: []string{"get_pages"},
		},
		{
			name:          "two tool calls",
			choices:       []*llms.ContentChoice{toolBlock("t1", "get_pages", `{}`), toolBlock("t2", "search", `{}`)},
			wantToolCalls: []string{"get_pages", "search"},
		},
		{
			name: "thinking, text and tool use",
			choices: []*llms.ContentChoice{
				thinkingBlock("plan it out"),
				textBlock("fetching"),
				toolBlock("t1", "get_pages", `{}`),
			},
			wantContent:   "fetching",
			wantReasoning: "plan it out",
			wantToolCalls: []string{"get_pages"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fm := newFake(&llms.ContentResponse{Choices: tc.choices})
			a := NewAdapter(fm, llmgate.ProviderAnthropic, "claude-sonnet-4-5", true)

			resp, err := a.Complete(context.Background(), llmgate.Request{
				Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
			})
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}

			if resp.Content != tc.wantContent {
				t.Errorf("Content = %q, want %q", resp.Content, tc.wantContent)
			}
			if resp.ReasoningContent != tc.wantReasoning {
				t.Errorf("ReasoningContent = %q, want %q", resp.ReasoningContent, tc.wantReasoning)
			}
			if len(resp.ToolCalls) != len(tc.wantToolCalls) {
				t.Fatalf("got %d tool calls, want %d (%v)", len(resp.ToolCalls), len(tc.wantToolCalls), tc.wantToolCalls)
			}
			for i, want := range tc.wantToolCalls {
				if resp.ToolCalls[i].Name != want {
					t.Errorf("ToolCalls[%d].Name = %q, want %q", i, resp.ToolCalls[i].Name, want)
				}
			}
		})
	}
}

// TestUsageCountedOnceAcrossBlocks guards the subtle half of the fold:
// every content block carries a copy of the same response-level usage, so
// summing across blocks would multiply the reported bill by the block
// count.
func TestUsageCountedOnceAcrossBlocks(t *testing.T) {
	fm := newFake(&llms.ContentResponse{Choices: []*llms.ContentChoice{
		thinkingBlock("thinking"),
		textBlock("answer"),
		toolBlock("t1", "search", `{}`),
	}})
	a := NewAdapter(fm, llmgate.ProviderAnthropic, "claude-sonnet-4-5", true)

	resp, err := a.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if resp.Usage.InputTokens != 1000 {
		t.Errorf("InputTokens = %d, want 1000 (3 blocks must not triple it)", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 200 {
		t.Errorf("OutputTokens = %d, want 200 (3 blocks must not triple it)", resp.Usage.OutputTokens)
	}
	if resp.Usage.TotalTokens != 1200 {
		t.Errorf("TotalTokens = %d, want 1200", resp.Usage.TotalTokens)
	}
}

// TestFoldPrefersChoiceWithUsage: a leading block without usage keys must
// not shadow a later block that has them.
func TestFoldPrefersChoiceWithUsage(t *testing.T) {
	noUsage := &llms.ContentChoice{Content: "partial", GenerationInfo: map[string]any{"Citations": "x"}}
	fm := newFake(&llms.ContentResponse{Choices: []*llms.ContentChoice{noUsage, textBlock(" rest")}})
	a := NewAdapter(fm, llmgate.ProviderAnthropic, "claude-sonnet-4-5", true)

	resp, err := a.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "partial rest" {
		t.Errorf("Content = %q, want %q", resp.Content, "partial rest")
	}
	if resp.Usage.InputTokens != 1000 {
		t.Errorf("InputTokens = %d, want 1000 — usage must be picked up from a later block", resp.Usage.InputTokens)
	}
}
