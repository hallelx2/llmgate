package pricing

import (
	"regexp"
	"strings"
)

// family groups models that share cache-pricing conventions.
type family int

const (
	familyUnknown family = iota
	familyAnthropic
	familyOpenAI
	familyGoogle
	// familyZhipu covers the GLM models. It exists to identify Zhipu's
	// own API among the many resellers that carry GLM (see provider.go);
	// it deliberately has no entry in familyCacheMultipliers, so GLM keeps
	// falling through to the conservative unknown-family default.
	familyZhipu
)

// vendorPrefixes are the leading segments gateways and SDKs bolt onto a
// model ID. Bedrock uses dots ("anthropic.claude-..."), everyone else
// uses slashes ("z-ai/glm-4.6", "models/gemini-2.5-flash").
var vendorPrefixes = []string{
	"anthropic.", "amazon.", "meta.", "mistral.", "cohere.", "ai21.",
}

// regionPrefixes are Bedrock's cross-region inference prefixes, which sit
// in front of the vendor prefix: "us.anthropic.claude-sonnet-4-5-v1:0".
var regionPrefixes = []string{"us.", "eu.", "apac.", "us-gov."}

var (
	// dateSuffix matches the snapshot dates providers pin releases with:
	// "-20250929" and "-2024-11-20".
	dateSuffix = regexp.MustCompile(`-(\d{8}|\d{4}-\d{2}-\d{2})$`)
	// versionSuffix matches Bedrock-style revisions: "-v1", "-v2:0".
	versionSuffix = regexp.MustCompile(`-v\d+$`)
)

// Canonical reduces a provider- or gateway-qualified model ID to the base
// ID the price book is keyed by.
//
//	claude-sonnet-4-5-20250929      -> claude-sonnet-4-5
//	models/gemini-2.5-flash         -> gemini-2.5-flash
//	z-ai/glm-4.6                    -> glm-4.6
//	us.anthropic.claude-opus-4-1-v1:0 -> claude-opus-4-1
//
// It never consults the price table, so it is safe to call on an unknown
// ID; the result is simply the best-effort base form.
func Canonical(model string) string {
	s := strings.ToLower(strings.TrimSpace(model))
	if s == "" {
		return ""
	}

	// Bedrock qualifies the revision with a colon: "...-v1:0".
	if i := strings.LastIndex(s, ":"); i > 0 {
		s = s[:i]
	}

	// A slash always separates a namespace from the model, so the last
	// segment is the ID: "openai/gpt-4o", "models/gemini-2.5-flash".
	if i := strings.LastIndex(s, "/"); i >= 0 {
		s = s[i+1:]
	}

	// Dotted vendor prefixes, optionally behind a region prefix. Only
	// strip a known vendor — model IDs legitimately contain dots
	// ("gpt-4.1-nano", "glm-4.6") and must not be truncated.
	for _, r := range regionPrefixes {
		if strings.HasPrefix(s, r) {
			s = strings.TrimPrefix(s, r)
			break
		}
	}
	for _, v := range vendorPrefixes {
		if strings.HasPrefix(s, v) {
			s = strings.TrimPrefix(s, v)
			break
		}
	}

	s = strings.TrimSuffix(s, "-latest")

	// Version before date, and it matters. Bedrock stacks both —
	// "claude-sonnet-4-5-20250929-v1:0" — and stripping the date first
	// finds no match, because "-v1" is in the way. That left the dated
	// Bedrock IDs on a different canonical key from the plain dated ID,
	// so a lookup for "claude-sonnet-4-5-20250929" landed in a bucket
	// holding only regional and GovCloud rates and quietly billed the
	// 10-20% premium.
	s = versionSuffix.ReplaceAllString(s, "")
	s = dateSuffix.ReplaceAllString(s, "")

	return s
}

// longestPrefix finds the longest key in table that prefixes canon at a
// segment boundary.
//
// Longest wins, and it has to: "claude-sonnet-4" also prefixes
// "claude-sonnet-4-5-preview", and picking it would bill Sonnet 4.5 at
// Sonnet 4 rates. The boundary check stops "glm-4.5" from matching
// "glm-4.55", which is a different model.
func longestPrefix(table map[string]Price, canon string) (string, bool) {
	best := ""
	for key := range table {
		if len(key) <= len(best) || !strings.HasPrefix(canon, key) {
			continue
		}
		// The match must end on a segment boundary, not mid-token.
		if rest := canon[len(key):]; rest != "" && rest[0] != '-' && rest[0] != '.' {
			continue
		}
		best = key
	}
	return best, best != ""
}

// familyOf classifies a model by its canonical ID, for the cache-rate
// fallbacks. Classification is by ID, not by the provider that served it —
// GLM reached through an Anthropic-compatible gateway is still GLM.
func familyOf(model string) family {
	s := Canonical(model)
	switch {
	case strings.HasPrefix(s, "claude"):
		return familyAnthropic
	case strings.HasPrefix(s, "gpt"), strings.HasPrefix(s, "chatgpt"),
		strings.HasPrefix(s, "o1"), strings.HasPrefix(s, "o3"), strings.HasPrefix(s, "o4"):
		return familyOpenAI
	case strings.HasPrefix(s, "gemini"):
		return familyGoogle
	case strings.HasPrefix(s, "glm"), strings.HasPrefix(s, "autoglm"):
		return familyZhipu
	default:
		return familyUnknown
	}
}
