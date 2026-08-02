package adapter

import (
	"context"
	"testing"

	"github.com/tmc/langchaingo/llms"

	"github.com/hallelx2/llmgate"
)

// usageFor runs one Complete against a response carrying the given
// GenerationInfo and returns the normalized Usage.
func usageFor(t *testing.T, model string, provider llmgate.Provider, gi map[string]any) llmgate.Usage {
	t.Helper()
	fm := newFake(&llms.ContentResponse{Choices: []*llms.ContentChoice{{
		Content:        "answer text",
		StopReason:     "stop",
		GenerationInfo: gi,
	}}})
	a := NewAdapter(fm, provider, model, true)

	resp, err := a.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "a fairly ordinary prompt"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	return resp.Usage
}

// TestAnthropicCacheTokensAreDisjoint: Anthropic reports cache creation
// and cache reads *alongside* input_tokens, so input must be left alone.
// Subtracting here would undercount the uncached prompt.
func TestAnthropicCacheTokensAreDisjoint(t *testing.T) {
	u := usageFor(t, "claude-sonnet-4-5", llmgate.ProviderAnthropic, map[string]any{
		"InputTokens":              500,
		"OutputTokens":             100,
		"CacheCreationInputTokens": 2000,
		"CacheReadInputTokens":     8000,
	})

	if u.InputTokens != 500 {
		t.Errorf("InputTokens = %d, want 500 — Anthropic already reports these disjoint", u.InputTokens)
	}
	if u.CacheWriteTokens != 2000 {
		t.Errorf("CacheWriteTokens = %d, want 2000", u.CacheWriteTokens)
	}
	if u.CacheReadTokens != 8000 {
		t.Errorf("CacheReadTokens = %d, want 8000", u.CacheReadTokens)
	}
	if u.TotalTokens != 10600 {
		t.Errorf("TotalTokens = %d, want 10600", u.TotalTokens)
	}
	if !u.TokensReported || u.Estimated {
		t.Errorf("TokensReported=%v Estimated=%v, want true/false", u.TokensReported, u.Estimated)
	}
}

// TestAnthropicCacheWriteIsBilled is the money regression: cache tokens
// used to be dropped, so a heavily cached call reported a cost far below
// the invoice.
func TestAnthropicCacheWriteIsBilled(t *testing.T) {
	cached := usageFor(t, "claude-sonnet-4-5", llmgate.ProviderAnthropic, map[string]any{
		"InputTokens":              0,
		"OutputTokens":             0,
		"CacheCreationInputTokens": 1_000_000,
	})
	if cached.CostUSD <= 0 {
		t.Fatal("a million cache-write tokens must not cost $0")
	}
	// Sonnet 4.5 cache write is 3.75/Mtok.
	if got, want := cached.CostUSD, 3.75; got < want*0.99 || got > want*1.01 {
		t.Errorf("CostUSD = %v, want ~%v", got, want)
	}
}

// TestOpenAICachedTokensSubtracted: OpenAI folds cached tokens *into*
// prompt_tokens, so input must have them removed or the cached portion
// gets billed twice — once at full rate, once at the cache rate.
func TestOpenAICachedTokensSubtracted(t *testing.T) {
	u := usageFor(t, "gpt-4o", llmgate.ProviderOpenAI, map[string]any{
		"PromptTokens":       10000, // includes the 8000 cached
		"CompletionTokens":   200,
		"PromptCachedTokens": 8000,
	})

	if u.InputTokens != 2000 {
		t.Errorf("InputTokens = %d, want 2000 (10000 prompt less 8000 cached)", u.InputTokens)
	}
	if u.CacheReadTokens != 8000 {
		t.Errorf("CacheReadTokens = %d, want 8000", u.CacheReadTokens)
	}
	// The prompt is still 10000 tokens in total, just billed in two tiers.
	if u.TotalTokens != 10200 {
		t.Errorf("TotalTokens = %d, want 10200", u.TotalTokens)
	}
}

// TestGoogleNonCachedInputPreferred: Google publishes the already
// subtracted figure, which is more trustworthy than our arithmetic.
func TestGoogleNonCachedInputPreferred(t *testing.T) {
	u := usageFor(t, "gemini-2.5-flash", llmgate.ProviderGemini, map[string]any{
		"input_tokens":         10000,
		"output_tokens":        300,
		"CachedTokens":         6000,
		"NonCachedInputTokens": 4000,
	})

	if u.InputTokens != 4000 {
		t.Errorf("InputTokens = %d, want 4000 from NonCachedInputTokens", u.InputTokens)
	}
	if u.CacheReadTokens != 6000 {
		t.Errorf("CacheReadTokens = %d, want 6000", u.CacheReadTokens)
	}
}

// TestReasoningTokensCaptured: reasoning is reported for visibility as a
// subset of output, so it must not inflate the total.
func TestReasoningTokensCaptured(t *testing.T) {
	u := usageFor(t, "o3", llmgate.ProviderOpenAI, map[string]any{
		"PromptTokens":     1000,
		"CompletionTokens": 900,
		"ReasoningTokens":  600,
	})

	if u.ReasoningTokens != 600 {
		t.Errorf("ReasoningTokens = %d, want 600", u.ReasoningTokens)
	}
	if u.OutputTokens != 900 {
		t.Errorf("OutputTokens = %d, want 900 — reasoning is a subset, not an addition", u.OutputTokens)
	}
	if u.TotalTokens != 1900 {
		t.Errorf("TotalTokens = %d, want 1900", u.TotalTokens)
	}
}

// TestMissingUsageIsEstimatedNotZero is the HAL-531 regression. A provider
// that reports no usage previously produced Priced:true with CostUSD:0 —
// a positive assertion that a call which returned content was free.
func TestMissingUsageIsEstimatedNotZero(t *testing.T) {
	u := usageFor(t, "claude-sonnet-4-5", llmgate.ProviderAnthropic, map[string]any{
		"SomeOtherMetadata": "no token counts here",
	})

	if u.TokensReported {
		t.Error("TokensReported must be false when the provider reported nothing")
	}
	if !u.Estimated {
		t.Error("Estimated must be true when counts came from the local tokenizer")
	}
	if u.InputTokens <= 0 || u.OutputTokens <= 0 {
		t.Errorf("estimated counts must be positive, got in=%d out=%d", u.InputTokens, u.OutputTokens)
	}
	if u.CostUSD <= 0 {
		t.Error("an estimated call must carry a non-zero cost — zero would read as free")
	}
	if !u.Priced {
		t.Error("Priced should stay true: a rate was found, only the counts were estimated")
	}
}

// TestGenuinelyZeroUsageStaysReported: a provider that really does report
// zeroes must not be mistaken for one that reported nothing.
func TestGenuinelyZeroUsageStaysReported(t *testing.T) {
	u := usageFor(t, "claude-sonnet-4-5", llmgate.ProviderAnthropic, map[string]any{
		"InputTokens":  0,
		"OutputTokens": 0,
	})

	if !u.TokensReported {
		t.Error("an explicit zero is still a report — TokensReported must be true")
	}
	if u.Estimated {
		t.Error("Estimated must be false when the provider actually answered")
	}
	if u.CostUSD != 0 {
		t.Errorf("CostUSD = %v, want 0 for a genuinely zero-token call", u.CostUSD)
	}
}

// TestUnreportedUsageWarnsOncePerModel keeps the warning useful under the
// engine's call volume — one line per model, not one per call.
func TestUnreportedUsageWarnsOncePerModel(t *testing.T) {
	var seen []string
	UnreportedUsageFunc = func(p llmgate.Provider, model string) {
		seen = append(seen, string(p)+"/"+model)
	}
	t.Cleanup(func() { UnreportedUsageFunc = nil })

	const model = "warn-once-test-model"
	for range 3 {
		usageFor(t, model, llmgate.ProviderAnthropic, map[string]any{"nothing": true})
	}

	if len(seen) != 1 {
		t.Fatalf("warned %d times %v, want exactly once per provider/model", len(seen), seen)
	}
}

// TestGLMUsageThroughAnthropicGateway: the production path. GLM is served
// by z.ai over the Anthropic protocol, so it arrives with Anthropic's key
// names but a GLM model ID, and must price off the GLM rates.
func TestGLMUsageThroughAnthropicGateway(t *testing.T) {
	u := usageFor(t, "glm-4.6", llmgate.ProviderAnthropic, map[string]any{
		"InputTokens":  1_000_000,
		"OutputTokens": 1_000_000,
	})

	if !u.Priced {
		t.Fatal("glm-4.6 must price even though it is served over the Anthropic driver")
	}
	// GLM-4.6: 0.60 in + 2.20 out.
	if got, want := u.CostUSD, 2.80; got < want*0.99 || got > want*1.01 {
		t.Errorf("CostUSD = %v, want ~%v at GLM rates (not Claude rates)", got, want)
	}
}
