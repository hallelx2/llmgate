package pricing_test

import (
	"math"
	"testing"

	"github.com/hallelx2/llmgate/pricing"
)

// TestCanonical covers the model-ID shapes that actually reach a price
// lookup. Before normalization every one of these missed the table and
// silently priced at $0.
func TestCanonical(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		// Dated snapshots — the IDs the APIs actually return.
		{"claude-sonnet-4-5-20250929", "claude-sonnet-4-5"},
		{"claude-sonnet-4-20250514", "claude-sonnet-4"},
		{"gpt-4o-2024-11-20", "gpt-4o"},

		// SDK and gateway prefixes.
		{"models/gemini-2.5-flash", "gemini-2.5-flash"},
		{"openai/gpt-4o", "gpt-4o"},
		{"z-ai/glm-4.6", "glm-4.6"},
		{"zhipuai/glm-4.6", "glm-4.6"},

		// Bedrock, with and without a cross-region prefix.
		{"anthropic.claude-sonnet-4-5-v1:0", "claude-sonnet-4-5"},
		{"us.anthropic.claude-opus-4-1-v1:0", "claude-opus-4-1"},

		// Aliases and casing.
		{"claude-haiku-4-5-latest", "claude-haiku-4-5"},
		{"GPT-4o", "gpt-4o"},
		{"  glm-4.6  ", "glm-4.6"},

		// Already canonical — must round-trip untouched. Dots inside a
		// model ID are not vendor prefixes and must survive.
		{"gpt-4.1-nano", "gpt-4.1-nano"},
		{"glm-4.5-air", "glm-4.5-air"},
		{"", ""},
	}

	for _, tc := range tests {
		if got := pricing.Canonical(tc.in); got != tc.want {
			t.Errorf("Canonical(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestLookupResolvesVariants is the end-to-end version: every variant must
// reach the same price as its base ID.
func TestLookupResolvesVariants(t *testing.T) {
	base, ok := pricing.Lookup("claude-sonnet-4-5")
	if !ok {
		t.Fatal("claude-sonnet-4-5 must be priced")
	}

	for _, id := range []string{
		"claude-sonnet-4-5-20250929",
		"anthropic.claude-sonnet-4-5-v1:0",
		"us.anthropic.claude-sonnet-4-5-v1:0",
		"anthropic/claude-sonnet-4-5",
	} {
		got, ok := pricing.Lookup(id)
		if !ok {
			t.Errorf("Lookup(%q) missed — it would price at $0", id)
			continue
		}
		if got != base {
			t.Errorf("Lookup(%q) = %+v, want %+v", id, got, base)
		}
	}
}

// TestLongestPrefixWins is the trap in prefix matching: a shorter table
// key can prefix a longer one ("claude-sonnet-4" prefixes
// "claude-sonnet-4-5-preview"), and picking it would bill the newer model
// at the older model's rate.
//
// The real Sonnet 4 and 4.5 happen to be priced identically, so this
// registers a synthetic pair with distinct rates — otherwise the
// assertion could not tell a correct resolution from a wrong one.
func TestLongestPrefixWins(t *testing.T) {
	cheap := pricing.Price{InputPerMTok: 1, OutputPerMTok: 2}
	dear := pricing.Price{InputPerMTok: 50, OutputPerMTok: 100}
	pricing.Register("prefixtest-model-3", cheap)
	pricing.Register("prefixtest-model-3-5", dear)

	got, ok := pricing.Lookup("prefixtest-model-3-5-preview")
	if !ok {
		t.Fatal("prefixtest-model-3-5-preview should resolve by prefix")
	}
	if got != dear {
		t.Errorf("Lookup resolved to %+v, want the longer key's price %+v — "+
			"a shorter prefix must never win", got, dear)
	}

	// And the shorter key still resolves its own variants.
	if got, _ := pricing.Lookup("prefixtest-model-3-preview"); got != cheap {
		t.Errorf("Lookup(prefixtest-model-3-preview) = %+v, want %+v", got, cheap)
	}
}

// TestPrefixRespectsBoundaries: a prefix match must not run into the
// middle of a token. "glm-4.5" must not absorb a hypothetical "glm-4.55".
func TestPrefixRespectsBoundaries(t *testing.T) {
	if _, ok := pricing.Lookup("glm-4.55-turbo"); ok {
		t.Error("glm-4.55-turbo must not match glm-4.5 — different model")
	}
	if _, ok := pricing.Lookup("gpt-4omini"); ok {
		t.Error("gpt-4omini must not match gpt-4o — no segment boundary")
	}
}

// TestRegisterBeatsNormalization: an explicitly registered ID is matched
// exactly, so a caller can always pin a rate the rules would otherwise
// resolve elsewhere.
func TestRegisterBeatsNormalization(t *testing.T) {
	const id = "claude-sonnet-4-5-20250929"
	pinned := pricing.Price{InputPerMTok: 99, OutputPerMTok: 199}
	pricing.Register(id, pinned)

	got, ok := pricing.Lookup(id)
	if !ok || got != pinned {
		t.Fatalf("Lookup(%q) = %+v ok=%v, want the pinned %+v", id, got, ok, pinned)
	}

	// The base ID must be unaffected by the pin.
	if base, _ := pricing.Lookup("claude-sonnet-4-5"); base == pinned {
		t.Error("pinning a dated ID must not overwrite the base entry")
	}
}

// TestGLMVariantsPriced guards our production model. GLM reaches the
// engine through an Anthropic-compatible gateway, and the aggregate price
// sources key it under vendor-prefixed names.
func TestGLMVariantsPriced(t *testing.T) {
	for _, id := range []string{"glm-4.6", "z-ai/glm-4.6", "zhipuai/glm-4.6", "glm-4.6-latest"} {
		p, ok := pricing.Lookup(id)
		if !ok {
			t.Errorf("Lookup(%q) missed — GLM is the model vectorless runs in production", id)
			continue
		}
		if p.InputPerMTok <= 0 || p.OutputPerMTok <= 0 {
			t.Errorf("Lookup(%q) returned non-positive rates %+v", id, p)
		}
	}
}

func TestUnknownModelStillUnpriced(t *testing.T) {
	if _, ok := pricing.Lookup("totally-made-up-model-zzz"); ok {
		t.Error("an unknown model must stay unpriced rather than prefix-matching something")
	}
}

// --- tiered cost (HAL-529) ----------------------------------------------

func approx(t *testing.T, got, want float64, label string) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("%s = %v, want %v", label, got, want)
	}
}

// TestComputeTokensCharges verifies each tier bills at its own rate rather
// than everything collapsing onto input/output.
func TestComputeTokensCharges(t *testing.T) {
	// claude-haiku-4-5: 1.00 in / 5.00 out / 1.25 cache-write / 0.10 cache-read
	cost, ok := pricing.ComputeTokens("claude-haiku-4-5", pricing.Tokens{
		Input:      1_000_000,
		Output:     1_000_000,
		CacheWrite: 1_000_000,
		CacheRead:  1_000_000,
	})
	if !ok {
		t.Fatal("claude-haiku-4-5 must be priced")
	}
	approx(t, cost, 1.00+5.00+1.25+0.10, "tiered cost")
}

// TestCacheWriteIsNotFree is the regression that matters: cache tokens
// used to be dropped entirely, so a cached prompt cost $0.
func TestCacheWriteIsNotFree(t *testing.T) {
	cost, _ := pricing.ComputeTokens("claude-sonnet-4-5", pricing.Tokens{CacheWrite: 100_000})
	if cost <= 0 {
		t.Fatal("cache-write tokens must be billed — they cost 1.25x input, not nothing")
	}
	// 100k at 3.75/Mtok
	approx(t, cost, 0.375, "cache-write cost")
}

// TestCacheRateFallbacksAreFamilySpecific: a table entry without explicit
// cache rates must fall back to its own family's multiplier, not one
// global guess. Anthropic reads at 0.1x, OpenAI at 0.5x, Google at 0.25x.
func TestCacheRateFallbacksAreFamilySpecific(t *testing.T) {
	tests := []struct {
		model    string
		input    float64
		wantMul  float64
		wantName string
	}{
		{"claude-fallback-test", 10.0, 0.10, "anthropic"},
		{"gpt-fallback-test", 10.0, 0.50, "openai"},
		{"gemini-fallback-test", 10.0, 0.25, "google"},
	}

	for _, tc := range tests {
		// Register with input/output only — no explicit cache rates.
		pricing.Register(tc.model, pricing.Price{InputPerMTok: tc.input, OutputPerMTok: tc.input * 4})

		cost, ok := pricing.ComputeTokens(tc.model, pricing.Tokens{CacheRead: 1_000_000})
		if !ok {
			t.Fatalf("%s must be priced", tc.model)
		}
		approx(t, cost, tc.input*tc.wantMul, tc.wantName+" cache-read fallback")
	}
}

// TestReasoningNotDoubleCharged: reasoning tokens are a subset of output.
// With no separate reasoning rate they must bill exactly as output.
func TestReasoningNotDoubleCharged(t *testing.T) {
	withReasoning, _ := pricing.ComputeTokens("o3", pricing.Tokens{Output: 1_000_000, Reasoning: 400_000})
	plain, _ := pricing.ComputeTokens("o3", pricing.Tokens{Output: 1_000_000})
	approx(t, withReasoning, plain, "reasoning subset of output")
}

// TestReasoningSeparateRate: when a model does carry its own reasoning
// rate, the reasoning portion is carved out of output rather than added.
func TestReasoningSeparateRate(t *testing.T) {
	pricing.Register("reasoning-rate-test", pricing.Price{
		InputPerMTok: 1, OutputPerMTok: 10, ReasoningPerMTok: 30,
	})
	cost, _ := pricing.ComputeTokens("reasoning-rate-test", pricing.Tokens{
		Output: 1_000_000, Reasoning: 400_000,
	})
	// 600k output at 10 + 400k reasoning at 30
	approx(t, cost, 0.6*10+0.4*30, "split reasoning cost")
}

// TestUnpricedStaysUnknown: the unpriced path must keep reporting
// (0, false) so callers can tell "unknown" from "free".
func TestUnpricedStaysUnknown(t *testing.T) {
	cost, ok := pricing.ComputeTokens("nonexistent-model-tiered-zzz", pricing.Tokens{Input: 1000, Output: 1000})
	if ok {
		t.Error("unknown model must report priced=false")
	}
	if cost != 0 {
		t.Errorf("unknown model cost = %v, want 0", cost)
	}
}

// TestDeprecatedComputeStillWorks: the old two-int helpers stay behaviour
// compatible so existing callers keep compiling and computing.
func TestDeprecatedComputeStillWorks(t *testing.T) {
	got := pricing.Compute("gpt-4o-mini", 1000, 500)
	approx(t, got, 0.00045, "legacy Compute")

	cost, ok := pricing.ComputeWithOK("gpt-4o-mini", 1000, 500)
	if !ok {
		t.Fatal("gpt-4o-mini must be priced")
	}
	approx(t, cost, 0.00045, "legacy ComputeWithOK")
}
