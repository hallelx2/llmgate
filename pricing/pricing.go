// Package pricing maintains a price table for known LLM models and
// computes per-call USD cost from token counts.
//
// Rates are keyed by model ID alone, never by provider. That is
// deliberate: a model is often served through a gateway that speaks
// another vendor's protocol — vectorless runs GLM-4.6 through an
// Anthropic-compatible endpoint — so a provider-scoped key would look up
// "anthropic/glm-4.6" and miss.
package pricing

import (
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Price is the USD cost per 1,000,000 tokens for a given model.
//
// Only Input and Output are required. The cache and reasoning rates fall
// back to family-derived multiples of the input rate when left at zero,
// so a table entry that predates cache pricing still computes something
// defensible rather than charging nothing.
type Price struct {
	InputPerMTok  float64
	OutputPerMTok float64

	// CacheWritePerMTok is the rate for tokens written into the prompt
	// cache. Zero falls back to InputPerMTok times the family's write
	// multiplier.
	CacheWritePerMTok float64

	// CacheReadPerMTok is the rate for tokens served from the prompt
	// cache. Zero falls back to InputPerMTok times the family's read
	// multiplier.
	CacheReadPerMTok float64

	// ReasoningPerMTok is the rate for thinking tokens. Zero means they
	// bill at OutputPerMTok, which is what every provider does today.
	ReasoningPerMTok float64
}

//go:generate go run ./gen

// The price book resolves in three layers, most authoritative first:
//
//  1. overrides  — explicit Register calls, always win
//  2. remote     — a refreshed snapshot, when one has been installed
//  3. defaults   — defaults_gen.go, compiled in and always available
//
// Every layer is optional except the last, and a lookup falls through on
// a miss. The point is that a remote refresh can never leave the library
// worse off than the embedded table it started with.
var priceMu sync.RWMutex
var overrides = map[string]Price{}

// remote is the installed remote snapshot, or nil. Read on every lookup,
// swapped wholesale by the refresher, so it is an atomic pointer rather
// than something guarded by priceMu.
var remote atomic.Pointer[snapshot]

// Lookup returns the price for a model, or (Price{}, false) if unknown.
//
// Within each layer the raw ID is tried before the canonical form, so a
// dated ID like "claude-sonnet-4-5-20250929", a prefixed one like
// "models/gemini-2.5-flash", or a gateway-qualified one like
// "z-ai/glm-4.6" resolves without needing its own entry.
func Lookup(model string) (Price, bool) {
	priceMu.RLock()
	p, ok := lookupIn(overrides, model)
	priceMu.RUnlock()
	if ok {
		return p, true
	}

	if snap := remote.Load(); snap != nil {
		if p, ok := lookupIn(snap.prices, model); ok {
			return p, true
		}
	}

	return lookupIn(defaultPrices, model)
}

// lookupIn resolves a model against one layer: exact match first so an
// explicitly keyed ID always wins, then the canonical form, then the
// longest matching prefix.
func lookupIn(table map[string]Price, model string) (Price, bool) {
	if p, ok := table[model]; ok {
		return p, true
	}
	canon := Canonical(model)
	if canon == "" {
		return Price{}, false
	}
	if p, ok := table[canon]; ok {
		return p, true
	}
	key, ok := longestPrefix(table, canon)
	if !ok {
		return Price{}, false
	}
	return table[key], true
}

// Register overrides or adds a price. Safe for init() in callers.
//
// Registered IDs match exactly, before any normalization, and sit above
// both the remote snapshot and the embedded table — so this is how to pin
// a rate that would otherwise resolve elsewhere.
func Register(model string, p Price) {
	priceMu.Lock()
	defer priceMu.Unlock()
	overrides[model] = p
}

// Unregister removes a previously registered override, so the model
// resolves through the remote snapshot and embedded table again. Removing
// an ID that was never registered is a no-op.
func Unregister(model string) {
	priceMu.Lock()
	defer priceMu.Unlock()
	delete(overrides, model)
}

// WarnFunc is invoked, when non-nil, once per distinct model that has no
// price-book entry, the first time a cost is computed for it. Wire it to a
// logger to surface "$0 because the model is unpriced" — otherwise an
// unpriced model is silently accounted as free, which is the failure mode
// that makes a quality-per-dollar benchmark read ∞. Safe to leave nil; the
// pricing package itself takes no logging dependency.
var WarnFunc func(model string)

var unpricedSeen sync.Map // model string -> struct{}; bounds WarnFunc to once per model

// Tokens is the token breakdown one call consumed, as the cost
// computation needs it. The fields are disjoint: Input excludes anything
// cached, so Input + CacheWrite + CacheRead is the whole prompt.
type Tokens struct {
	Input      int
	Output     int
	CacheWrite int
	CacheRead  int
	// Reasoning is a subset of Output, not an addition. It only changes
	// the bill when the model prices reasoning separately.
	Reasoning int
}

// ComputeTokens returns the USD cost for a token breakdown at the model's
// rates, and whether the model was actually priced.
//
// A (0, false) result means "cost unknown" — not a free call. Callers
// reporting spend must keep the two apart.
func ComputeTokens(model string, tk Tokens) (float64, bool) {
	p, ok := Lookup(model)
	if !ok {
		if WarnFunc != nil {
			if _, loaded := unpricedSeen.LoadOrStore(model, struct{}{}); !loaded {
				WarnFunc(model)
			}
		}
		return 0, false
	}

	writeRate, readRate := cacheRates(model, p)

	// Reasoning is a subset of Output. Bill it separately only when the
	// model carries its own reasoning rate; otherwise the output rate
	// already covers it and splitting would double-charge.
	output, reasoning := tk.Output, 0
	if p.ReasoningPerMTok > 0 {
		reasoning = min(tk.Reasoning, tk.Output)
		output -= reasoning
	}

	total := float64(tk.Input)*p.InputPerMTok +
		float64(output)*p.OutputPerMTok +
		float64(reasoning)*p.ReasoningPerMTok +
		float64(tk.CacheWrite)*writeRate +
		float64(tk.CacheRead)*readRate

	return total / 1_000_000.0, true
}

// cacheRates resolves the cache write/read rates for a model, filling
// unset entries from the family's published multiple of the input rate.
//
// Estimating beats charging zero: an unset cache-write rate previously
// billed a cached prompt at nothing, understating the invoice by the
// entire cached prefix. Explicit table values always win.
func cacheRates(model string, p Price) (write, read float64) {
	write, read = p.CacheWritePerMTok, p.CacheReadPerMTok
	if write > 0 && read > 0 {
		return write, read
	}

	writeMul, readMul := familyCacheMultipliers(model)
	if write == 0 {
		write = p.InputPerMTok * writeMul
	}
	if read == 0 {
		read = p.InputPerMTok * readMul
	}
	return write, read
}

// familyCacheMultipliers returns a family's (write, read) cache rates as
// multiples of its input rate. The three families differ enough that one
// global default would be wrong for two of them.
func familyCacheMultipliers(model string) (write, read float64) {
	switch familyOf(model) {
	case familyAnthropic:
		return 1.25, 0.10
	case familyOpenAI:
		return 1.0, 0.50 // implicit caching: no write premium
	case familyGoogle:
		return 1.0, 0.25
	default:
		// Unknown family: assume no write premium and the least
		// generous read discount, so an unrecognised model is never
		// accounted cheaper than it really is.
		return 1.0, 0.50
	}
}

// ComputeWithOK returns the USD cost for the given input/output counts and
// whether the model was priced.
//
// Deprecated: use ComputeTokens, which also accounts for cache and
// reasoning tokens. This form bills a cached prompt as if it were
// uncached.
func ComputeWithOK(model string, in, out int) (float64, bool) {
	return ComputeTokens(model, Tokens{Input: in, Output: out})
}

// Compute returns the USD cost for the given token counts at the model's
// rate, or 0 if the model isn't priced.
//
// Deprecated: use ComputeTokens. This form discards the priced flag, so an
// unpriced model is indistinguishable from a free call.
func Compute(model string, in, out int) float64 {
	cost, _ := ComputeTokens(model, Tokens{Input: in, Output: out})
	return cost
}

// DefaultsAsOf reports when the compiled-in price table was generated from
// the upstream feed.
//
// AsOf tells you the vintage of the remote snapshot, and reports the zero
// time when there is none. This is the answer for the other case: how old
// the rates behind a cost figure are when nothing remote is installed.
func DefaultsAsOf() time.Time {
	t, err := time.Parse(time.RFC3339, defaultsAsOf)
	if err != nil {
		return time.Time{}
	}
	return t
}

// KnownModels returns every model ID with pricing data, across all three
// layers, deduplicated and sorted. Useful for diagnostics.
func KnownModels() []string {
	seen := map[string]struct{}{}

	priceMu.RLock()
	for k := range overrides {
		seen[k] = struct{}{}
	}
	priceMu.RUnlock()

	if snap := remote.Load(); snap != nil {
		for k := range snap.prices {
			seen[k] = struct{}{}
		}
	}
	for k := range defaultPrices {
		seen[k] = struct{}{}
	}

	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
