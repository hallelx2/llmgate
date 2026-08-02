package pricing_test

import (
	"testing"
	"time"

	"github.com/hallelx2/llmgate/pricing"
)

// TestFamilyNameIsNotAModel is the regression test for a 10x over-report.
//
// LiteLLM carries "anthropic.claude-v1" and "anthropic.claude-v2:1" at
// Claude 1's 8.00/24.00. Canonical stripped the vendor prefix and the "-v1",
// leaving a bare "claude" in the price book — and because longestPrefix
// matches at segment boundaries, that key then answered for every Claude
// model not listed explicitly. claude-haiku-3-5 came back 8.00/24.00
// instead of its real 0.80/4.00.
func TestFamilyNameIsNotAModel(t *testing.T) {
	for _, id := range []string{
		"anthropic.claude-v1",
		"anthropic.claude-v2:1",
		"bedrock/us-east-1/anthropic.claude-v1",
	} {
		if got := pricing.Canonical(id); got == "claude" {
			t.Errorf("Canonical(%q) = %q — a bare family name becomes a catch-all "+
				"prefix for every unlisted Claude model", id, got)
		}
	}

	// And the model that was mispriced by it.
	p, ok := pricing.Lookup("claude-haiku-3-5")
	if !ok {
		t.Fatal("claude-haiku-3-5 must be priced")
	}
	if p.InputPerMTok != 0.80 || p.OutputPerMTok != 4.00 {
		t.Errorf("claude-haiku-3-5 = %.4f/%.4f, want 0.8000/4.0000", p.InputPerMTok, p.OutputPerMTok)
	}
}

// TestGeneratedDefaultsAreSane spot-checks the generated table against
// vendor list prices. These are the rates the vendors publish; if a
// regeneration moves one, that is either a real price change worth noticing
// in review or a bug in the generator — both warrant a failing test.
func TestGeneratedDefaultsAreSane(t *testing.T) {
	want := map[string]struct{ in, out float64 }{
		"claude-opus-4-1":   {15.00, 75.00},
		"claude-sonnet-4-5": {3.00, 15.00},
		"claude-haiku-4-5":  {1.00, 5.00},
		"claude-haiku-3-5":  {0.80, 4.00},
		"gpt-4o":            {2.50, 10.00},
		"gpt-4o-mini":       {0.15, 0.60},
		"gemini-2.5-pro":    {1.25, 10.00},
		"gemini-2.5-flash":  {0.30, 2.50},
		"glm-4.6":           {0.60, 2.20},
		"glm-4.5-air":       {0.20, 1.10},
	}

	for id, w := range want {
		p, ok := pricing.Lookup(id)
		if !ok {
			t.Errorf("%s: not in the embedded table", id)
			continue
		}
		if p.InputPerMTok != w.in || p.OutputPerMTok != w.out {
			t.Errorf("%s = %.4f/%.4f, want %.4f/%.4f", id, p.InputPerMTok, p.OutputPerMTok, w.in, w.out)
		}
	}
}

// TestNoAbsurdDefaults guards the units error that would do the most
// damage. Every rate the vendors publish sits well inside this range, so
// anything outside it means the generator wrote per-token values as
// per-million or similar.
func TestNoAbsurdDefaults(t *testing.T) {
	for _, id := range pricing.KnownModels() {
		p, ok := pricing.Lookup(id)
		if !ok {
			continue
		}
		if p.InputPerMTok <= 0 || p.InputPerMTok > 200 {
			t.Errorf("%s: input %.4f per MTok is outside any plausible range", id, p.InputPerMTok)
		}
		if p.OutputPerMTok <= 0 || p.OutputPerMTok > 1000 {
			t.Errorf("%s: output %.4f per MTok is outside any plausible range", id, p.OutputPerMTok)
		}
		if p.CacheReadPerMTok > p.InputPerMTok {
			t.Errorf("%s: cache read %.4f exceeds input %.4f — a cache hit is never "+
				"more expensive than the uncached token", id, p.CacheReadPerMTok, p.InputPerMTok)
		}
	}
}

// TestDefaultsAsOfIsParseable keeps the provenance stamp honest: a caller
// reading it wants to know how old the rates behind a cost figure are.
func TestDefaultsAsOfIsParseable(t *testing.T) {
	got := pricing.DefaultsAsOf()
	if got.IsZero() {
		t.Fatal("DefaultsAsOf is zero — the generated stamp did not parse")
	}
	if got.After(time.Now().Add(24 * time.Hour)) {
		t.Errorf("DefaultsAsOf = %s, which is in the future", got)
	}
}
