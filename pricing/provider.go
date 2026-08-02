package pricing

import (
	"slices"
	"strings"
)

// Upstream price feeds list the same model many times over, once per place
// you can buy it. LiteLLM carries 64 GLM entries; six of them collapse onto
// the canonical ID "glm-4.6" alone:
//
//	zai/glm-4.6                     0.600 / 2.200   <- Zhipu's own API
//	together_ai/zai-org/GLM-4.6     0.600 / 2.200
//	novita/zai-org/glm-4.6          0.550 / 2.200
//	vercel_ai_gateway/zai/glm-4.6   0.450 / 1.800
//	openrouter/z-ai/glm-4.6         0.400 / 1.750
//	cerebras/zai-glm-4.6            2.250 / 2.750
//
// These are not disagreeing about one price. They are six different prices,
// each correct for whoever is invoicing you. Canonical() throws the
// namespace away, so without a rule the snapshot keeps whichever entry the
// map happened to visit last — a 5.6x spread decided by iteration order.
//
// The rule: the vendor that made the model wins. Anyone calling the model's
// own API — which is the common case, and the one vectorless is in with
// z.ai — then gets the rate they are actually billed. Resellers undercut or
// mark up the first party, so defaulting to one of them is wrong in an
// unpredictable direction, and quietly so.
type sourceRank int

const (
	// rankFirstParty is the vendor's own API: a bare "claude-sonnet-4-5",
	// or a vendor-owned namespace like "zai/glm-4.6".
	rankFirstParty sourceRank = iota
	// rankVendorCloud is the vendor's model on its own global cloud
	// endpoint — "global.anthropic.claude-...", "anthropic.claude-..." on
	// Bedrock. Same list price as first-party today.
	rankVendorCloud
	// rankRegional is a region-pinned cloud endpoint. Bedrock and Vertex
	// both charge a 10% premium over their global endpoints.
	rankRegional
	// rankGov is GovCloud, a further premium again.
	rankGov
	// rankReseller is everyone else: gateways, inference hosts, brokers.
	// Unknown namespaces land here deliberately — a namespace we do not
	// recognise is far more likely to be a new reseller than a vendor.
	rankReseller
)

// firstPartyNamespaces maps a model family to the upstream namespaces that
// mean "the vendor's own API".
//
// Feeds disagree on spelling for the same vendor — LiteLLM writes "zai",
// models.dev writes both "zai" and "zhipuai", OpenRouter writes "z-ai" —
// so each family accepts a set.
var firstPartyNamespaces = map[family][]string{
	familyAnthropic: {"anthropic", "claude"},
	familyOpenAI:    {"openai"},
	familyGoogle:    {"google", "gemini", "models"},
	familyZhipu:     {"zai", "zhipuai", "z-ai", "zhipu", "bigmodel"},
}

// vendorCloudNamespaces are the vendors' own managed clouds. Not the model
// maker, but not a reseller either: first-party-adjacent, list-priced, and
// the right answer when no truer entry exists.
var vendorCloudNamespaces = []string{"bedrock", "vertex_ai", "azure", "azure_ai"}

// govMarkers identify GovCloud IDs, which are premium-priced and must never
// win a collision against a commercial rate.
var govMarkers = []string{"us-gov", "usgov"}

// regionMarkers are the geographic qualifiers cloud platforms put on an ID,
// either as a namespace segment ("bedrock/eu-west-1/...") or as a dotted
// prefix ("eu.anthropic..."). They carry a premium over the global endpoint.
var regionMarkers = []string{
	"us-east", "us-west", "eu-west", "eu-central", "ap-south", "ap-northeast",
	"ap-southeast", "ca-central", "sa-east", "me-central", "af-south",
}

// dottedRegionPrefixes are Bedrock's cross-region inference prefixes.
// regionPrefixes in canonical.go covers what Canonical strips; this list is
// what ranking needs to *see*, so it includes the ones Canonical leaves
// alone ("au.", "jp.", "global.").
var dottedRegionPrefixes = []string{"us.", "eu.", "apac.", "au.", "jp.", "ca.", "sa."}

// rankOf scores how authoritative an upstream ID is as *the* price for its
// canonical model. Lower wins.
//
// It reads the original ID, not the canonical form, because the namespace
// and prefixes Canonical strips are exactly the evidence needed.
func rankOf(id string) sourceRank {
	s := strings.ToLower(strings.TrimSpace(id))
	if s == "" {
		return rankReseller
	}

	// GovCloud anywhere in the ID settles it — "us-gov.anthropic..." and
	// "bedrock/us-gov-east-1/..." are both premium-priced.
	for _, g := range govMarkers {
		if strings.Contains(s, g) {
			return rankGov
		}
	}

	namespace, last := splitNamespace(s)

	if namespace == "" {
		// No namespace: judge by the dotted prefixes instead.
		switch {
		case strings.HasPrefix(last, "global."):
			return rankVendorCloud
		case hasAnyPrefix(last, dottedRegionPrefixes):
			return rankRegional
		case hasAnyPrefix(last, vendorPrefixes):
			// "anthropic.claude-sonnet-4-5-v1:0" — a Bedrock ID that
			// happens to carry no region.
			return rankVendorCloud
		default:
			// A bare "claude-sonnet-4-5" or "gpt-4o": the vendor's own
			// catalogue entry, and the best evidence there is.
			return rankFirstParty
		}
	}

	// A namespaced ID. The first segment names who sells it.
	head := namespace
	if before, _, found := strings.Cut(namespace, "/"); found {
		head = before
	}

	if containsAny(namespace, regionMarkers) {
		return rankRegional
	}
	if fam := familyOf(s); slices.Contains(firstPartyNamespaces[fam], head) {
		return rankFirstParty
	}
	if slices.Contains(vendorCloudNamespaces, head) {
		return rankVendorCloud
	}
	return rankReseller
}

// splitNamespace divides an ID at its last slash: everything before is the
// namespace, everything after is the model segment. IDs without a slash
// have no namespace.
func splitNamespace(s string) (namespace, last string) {
	i := strings.LastIndex(s, "/")
	if i < 0 {
		return "", s
	}
	return s[:i], s[i+1:]
}

func hasAnyPrefix(s string, prefixes []string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

func containsAny(s string, subs []string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

// better reports whether candidate should replace incumbent as the entry
// for a canonical key.
//
// Rank decides it. When rank ties — two resellers, say — the
// lexicographically smaller ID wins. That tie-break carries no meaning; its
// only job is to make the result independent of map iteration order, so the
// same feed always produces the same snapshot.
func better(candidateID string, candidate Price, incumbentID string, incumbent Price) bool {
	cr, ir := rankOf(candidateID), rankOf(incumbentID)
	if cr != ir {
		return cr < ir
	}
	// Same rank and the incumbent carries cache rates the candidate lacks:
	// keep the richer entry rather than trading data for alphabetical order.
	if incumbent.CacheReadPerMTok > 0 && candidate.CacheReadPerMTok == 0 {
		return false
	}
	if candidate.CacheReadPerMTok > 0 && incumbent.CacheReadPerMTok == 0 {
		return true
	}
	return candidateID < incumbentID
}
