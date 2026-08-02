package pricing_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/hallelx2/llmgate/pricing"
)

// The IDs and rates below are copied from LiteLLM's live feed. They are the
// whole point of these tests: a fixture with one entry per canonical key
// cannot reproduce the bug, because the bug is in how collisions resolve.

// glmFeed is the six upstream entries that collapse onto "glm-4.6", plus
// the two that collapse onto "glm-4.5". Only zai/ is Zhipu's own API, and
// vectorless calls Zhipu directly, so zai/ is the rate it is billed at.
const glmFeed = `{
  "zai/glm-4.6":                   {"input_cost_per_token": 6.0e-07, "output_cost_per_token": 2.2e-06, "cache_read_input_token_cost": 1.1e-07, "mode": "chat"},
  "together_ai/zai-org/GLM-4.6":   {"input_cost_per_token": 6.0e-07, "output_cost_per_token": 2.2e-06, "mode": "chat"},
  "novita/zai-org/glm-4.6":        {"input_cost_per_token": 5.5e-07, "output_cost_per_token": 2.2e-06, "cache_read_input_token_cost": 1.1e-07, "mode": "chat"},
  "vercel_ai_gateway/zai/glm-4.6": {"input_cost_per_token": 4.5e-07, "output_cost_per_token": 1.8e-06, "cache_read_input_token_cost": 1.1e-07, "mode": "chat"},
  "openrouter/z-ai/glm-4.6":       {"input_cost_per_token": 4.0e-07, "output_cost_per_token": 1.75e-06, "mode": "chat"},
  "baseten/zai-org/GLM-4.6":       {"input_cost_per_token": 6.0e-07, "output_cost_per_token": 2.2e-06, "mode": "chat"},
  "zai/glm-4.5":                   {"input_cost_per_token": 6.0e-07, "output_cost_per_token": 2.2e-06, "mode": "chat"},
  "novita/zai-org/glm-4.5":        {"input_cost_per_token": 6.0e-07, "output_cost_per_token": 2.2e-06, "cache_read_input_token_cost": 1.1e-07, "mode": "chat"}
}`

// sonnetFeed is the Anthropic Sonnet 4.5 cluster: first-party, Bedrock
// global, three regional endpoints at a 10% premium, and GovCloud at 20%.
const sonnetFeed = `{
  "claude-sonnet-4-5":                              {"input_cost_per_token": 3.0e-06, "output_cost_per_token": 1.5e-05, "cache_read_input_token_cost": 3.0e-07, "mode": "chat"},
  "claude-sonnet-4-5-20250929":                     {"input_cost_per_token": 3.0e-06, "output_cost_per_token": 1.5e-05, "mode": "chat"},
  "global.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input_cost_per_token": 3.0e-06, "output_cost_per_token": 1.5e-05, "mode": "chat"},
  "us.anthropic.claude-sonnet-4-5-20250929-v1:0":   {"input_cost_per_token": 3.3e-06, "output_cost_per_token": 1.65e-05, "mode": "chat"},
  "eu.anthropic.claude-sonnet-4-5-20250929-v1:0":   {"input_cost_per_token": 3.3e-06, "output_cost_per_token": 1.65e-05, "mode": "chat"},
  "jp.anthropic.claude-sonnet-4-5-20250929-v1:0":   {"input_cost_per_token": 3.3e-06, "output_cost_per_token": 1.65e-05, "mode": "chat"},
  "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input_cost_per_token": 3.6e-06, "output_cost_per_token": 1.8e-05, "mode": "chat"},
  "bedrock/us-gov-west-1/claude-sonnet-4-5-20250929-v1:0": {"input_cost_per_token": 3.6e-06, "output_cost_per_token": 1.8e-05, "mode": "chat"}
}`

// withFeed installs a snapshot built from the given LiteLLM-shaped JSON and
// removes it when the test ends.
func withFeed(t *testing.T, body string) []string {
	t.Helper()
	srv := serveJSON(t, body)
	var errs []string
	stop, err := pricing.UseRemote(context.Background(), pricing.RemoteConfig{
		Sources:         []pricing.Source{pricing.LiteLLMSource{URL: srv.URL}},
		RefreshInterval: -1,
		HTTPClient:      srv.Client(),
		OnError:         func(src string, e error) { errs = append(errs, src+": "+e.Error()) },
	})
	if err != nil {
		t.Fatalf("UseRemote: %v", err)
	}
	t.Cleanup(stop)
	return errs
}

// TestZhipuFirstPartyWinsItsCollision is the regression test for the bug
// that motivated all of this. Before the precedence rule, glm-4.6 resolved
// to novita's 0.55 or vercel's 0.45 depending on map iteration order, and
// the rate vectorless is actually billed never won at all.
func TestZhipuFirstPartyWinsItsCollision(t *testing.T) {
	withFeed(t, glmFeed)

	p, ok := pricing.Lookup("glm-4.6")
	if !ok {
		t.Fatal("glm-4.6 must be priced")
	}
	if p.InputPerMTok != 0.60 || p.OutputPerMTok != 2.20 {
		t.Errorf("glm-4.6 = %.4f/%.4f, want Zhipu's own 0.6000/2.2000 — a reseller rate won the collision",
			p.InputPerMTok, p.OutputPerMTok)
	}
}

// TestGatewayQualifiedIDStillResolvesFirstParty covers the way vectorless
// actually names the model: it calls Zhipu through an Anthropic-compatible
// endpoint, so the ID reaching the price book carries a gateway prefix.
func TestGatewayQualifiedIDStillResolvesFirstParty(t *testing.T) {
	withFeed(t, glmFeed)

	for _, id := range []string{"glm-4.6", "z-ai/glm-4.6", "zai/glm-4.6", "anthropic/glm-4.6"} {
		p, ok := pricing.Lookup(id)
		if !ok {
			t.Errorf("%s: not priced", id)
			continue
		}
		if p.InputPerMTok != 0.60 {
			t.Errorf("%s input = %.4f, want 0.6000", id, p.InputPerMTok)
		}
	}
}

// TestFirstPartyBeatsRegionalAndGov pins the ordering among Anthropic's own
// endpoints. Regional carries a 10% premium and GovCloud 20%; neither is
// what a first-party API caller pays.
func TestFirstPartyBeatsRegionalAndGov(t *testing.T) {
	withFeed(t, sonnetFeed)

	for _, id := range []string{"claude-sonnet-4-5", "claude-sonnet-4-5-20250929"} {
		p, ok := pricing.Lookup(id)
		if !ok {
			t.Fatalf("%s must be priced", id)
		}
		if p.InputPerMTok != 3.00 || p.OutputPerMTok != 15.00 {
			t.Errorf("%s = %.4f/%.4f, want 3.0000/15.0000 — a regional or GovCloud rate won",
				id, p.InputPerMTok, p.OutputPerMTok)
		}
	}
}

// TestCollisionIsDeterministic is the test the original code could not have
// passed. Go randomizes map iteration per range, so installing the same
// feed repeatedly exercises many different orderings; the resolved rate has
// to be identical every time.
func TestCollisionIsDeterministic(t *testing.T) {
	const rounds = 60
	first := 0.0
	for i := range rounds {
		func() {
			srv := serveJSON(t, glmFeed)
			stop, err := pricing.UseRemote(context.Background(), pricing.RemoteConfig{
				Sources:         []pricing.Source{pricing.LiteLLMSource{URL: srv.URL}},
				RefreshInterval: -1,
				HTTPClient:      srv.Client(),
			})
			if err != nil {
				t.Fatalf("round %d: UseRemote: %v", i, err)
			}
			defer stop()

			p, _ := pricing.Lookup("glm-4.6")
			if i == 0 {
				first = p.InputPerMTok
				return
			}
			if p.InputPerMTok != first {
				t.Fatalf("round %d: glm-4.6 input = %.4f, round 0 gave %.4f — resolution depends on map order",
					i, p.InputPerMTok, first)
			}
		}()
	}
	if first != 0.60 {
		t.Errorf("glm-4.6 settled on %.4f, want 0.6000", first)
	}
}

// TestPoisonedEntryDoesNotCondemnSnapshot covers the live landmine:
// LiteLLM carries wandb/zai-org/GLM-4.5 at 55000/200000, an upstream units
// error that canonicalizes onto glm-4.5. Previously it could win its
// collision and fail the drift check, rejecting every other model in the
// feed along with it.
func TestPoisonedEntryDoesNotCondemnSnapshot(t *testing.T) {
	feed := fmt.Sprintf(`{
      "wandb/zai-org/GLM-4.5": {"input_cost_per_token": 5.5e-02, "output_cost_per_token": 2.0e-01, "mode": "chat"},
      %s`, glmFeed[1:])

	errs := withFeed(t, feed)

	// The good models must still be there.
	if p, ok := pricing.Lookup("glm-4.6"); !ok || p.InputPerMTok != 0.60 {
		t.Errorf("glm-4.6 = %.4f (priced=%v), want 0.6000 — one bad row must not sink the snapshot",
			p.InputPerMTok, ok)
	}
	// And glm-4.5 must not have taken the poisoned rate.
	if p, _ := pricing.Lookup("glm-4.5"); p.InputPerMTok != 0.60 {
		t.Errorf("glm-4.5 input = %.4f, want 0.6000 — the poisoned entry was adopted", p.InputPerMTok)
	}
	// The drop has to be visible, not silent.
	if len(errs) == 0 {
		t.Error("dropping an implausible entry must be reported through OnError")
	}
}

// TestCacheRatesSurviveASilentFeed covers the interaction between the
// precedence rule and missing data: "zai/glm-4.6" is the right entry on
// rank but omits the cache-write rate, and a reseller entry that happens to
// carry one must not be preferred just for being richer. The embedded
// table's published rate fills the gap instead.
func TestCacheRatesSurviveASilentFeed(t *testing.T) {
	withFeed(t, glmFeed)

	p, ok := pricing.Lookup("glm-4.5")
	if !ok {
		t.Fatal("glm-4.5 must be priced")
	}
	if p.InputPerMTok != 0.60 {
		t.Errorf("glm-4.5 input = %.4f, want Zhipu's 0.6000", p.InputPerMTok)
	}
	if p.CacheReadPerMTok != 0.11 {
		t.Errorf("glm-4.5 cache read = %.4f, want the embedded 0.1100 — a silent feed must not "+
			"erase a published rate and fall through to the family guess", p.CacheReadPerMTok)
	}
}
