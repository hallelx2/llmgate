package pricing

import "testing"

// TestRankOf pins the precedence ladder against real upstream IDs. Every
// entry here is copied from LiteLLM's live feed.
func TestRankOf(t *testing.T) {
	cases := []struct {
		id   string
		want sourceRank
	}{
		// Bare vendor catalogue IDs.
		{"claude-sonnet-4-5", rankFirstParty},
		{"claude-sonnet-4-5-20250929", rankFirstParty},
		{"gpt-4o", rankFirstParty},
		{"gemini-2.5-flash", rankFirstParty},

		// Vendor-owned namespaces.
		{"zai/glm-4.6", rankFirstParty},
		{"zai/glm-4.5-air", rankFirstParty},
		{"anthropic/claude-sonnet-4-5", rankFirstParty},
		{"openai/gpt-4o", rankFirstParty},
		{"gemini/gemini-2.5-flash", rankFirstParty},
		{"models/gemini-2.5-flash", rankFirstParty},

		// The vendor's own cloud, global endpoint.
		{"global.anthropic.claude-sonnet-4-5-20250929-v1:0", rankVendorCloud},
		{"anthropic.claude-sonnet-4-5-20250929-v1:0", rankVendorCloud},
		{"azure_ai/claude-sonnet-4-5", rankVendorCloud},

		// Region-pinned endpoints carry a premium.
		{"us.anthropic.claude-sonnet-4-5-20250929-v1:0", rankRegional},
		{"eu.anthropic.claude-sonnet-4-5-20250929-v1:0", rankRegional},
		{"jp.anthropic.claude-sonnet-4-5-20250929-v1:0", rankRegional},
		{"bedrock/us-east-1/zai.glm-5", rankRegional},

		// GovCloud, in either spelling.
		{"us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0", rankGov},
		{"bedrock/us-gov-west-1/claude-sonnet-4-5-20250929-v1:0", rankGov},

		// Everyone reselling someone else's model.
		{"novita/zai-org/glm-4.6", rankReseller},
		{"openrouter/z-ai/glm-4.6", rankReseller},
		{"together_ai/zai-org/GLM-4.6", rankReseller},
		{"vercel_ai_gateway/zai/glm-4.6", rankReseller},
		{"cerebras/zai-glm-4.6", rankReseller},
		{"baseten/zai-org/GLM-4.6", rankReseller},
		{"deepinfra/google/gemini-2.5-flash", rankReseller},
		{"wandb/zai-org/GLM-4.5", rankReseller},
		{"fireworks_ai/accounts/fireworks/models/glm-4p6", rankReseller},

		// An unrecognised namespace is assumed to be a reseller, because a
		// new gateway is far likelier than a new model vendor.
		{"some-new-gateway/glm-4.6", rankReseller},
	}

	for _, c := range cases {
		if got := rankOf(c.id); got != c.want {
			t.Errorf("rankOf(%q) = %d, want %d", c.id, got, c.want)
		}
	}
}

// TestBetterIsATotalOrder checks the tie-break actually breaks ties: equal
// rank and equal cache data must resolve by ID, so no pair is ambiguous.
func TestBetterIsATotalOrder(t *testing.T) {
	p := Price{InputPerMTok: 1, OutputPerMTok: 2}
	a, b := "novita/zai-org/glm-4.6", "together_ai/zai-org/GLM-4.6"

	if better(a, p, b, p) == better(b, p, a, p) {
		t.Fatalf("better() must order %q and %q strictly, got the same answer both ways", a, b)
	}
}

// TestBetterPrefersRicherEntryAtEqualRank keeps the original behaviour that
// a cache-bearing entry beats one without, when rank cannot separate them.
func TestBetterPrefersRicherEntryAtEqualRank(t *testing.T) {
	rich := Price{InputPerMTok: 1, OutputPerMTok: 2, CacheReadPerMTok: 0.1}
	poor := Price{InputPerMTok: 1, OutputPerMTok: 2}

	// "zzz/..." sorts last, so only the cache rate can win it this.
	if !better("zzz-gateway/glm-4.6", rich, "aaa-gateway/glm-4.6", poor) {
		t.Error("an entry carrying cache rates must beat one without at equal rank")
	}
	if better("aaa-gateway/glm-4.6", poor, "zzz-gateway/glm-4.6", rich) {
		t.Error("an entry without cache rates must not displace one carrying them")
	}
}
