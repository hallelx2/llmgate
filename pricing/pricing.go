// Package pricing maintains a static price table for known LLM models
// and computes per-call USD cost from token counts.
package pricing

import "sync"

// Price is the USD cost per 1,000,000 tokens for a given model.
type Price struct {
	InputPerMTok  float64
	OutputPerMTok float64
}

// Prices are public list prices as of April 2026. Refresh as providers update.
// Source: official pricing pages for Anthropic, OpenAI, and Google.
var defaultPrices = map[string]Price{
	// ── Anthropic ─────────────────────────────────────────────────
	"claude-sonnet-4-5":        {InputPerMTok: 3.00, OutputPerMTok: 15.00},
	"claude-sonnet-4-20250514": {InputPerMTok: 3.00, OutputPerMTok: 15.00},
	"claude-opus-4-1":          {InputPerMTok: 15.00, OutputPerMTok: 75.00},
	"claude-haiku-4-5":         {InputPerMTok: 1.00, OutputPerMTok: 5.00},
	"claude-haiku-3-5":         {InputPerMTok: 0.80, OutputPerMTok: 4.00},

	// ── OpenAI ────────────────────────────────────────────────────
	"gpt-4o":       {InputPerMTok: 2.50, OutputPerMTok: 10.00},
	"gpt-4o-mini":  {InputPerMTok: 0.15, OutputPerMTok: 0.60},
	"gpt-4.1":      {InputPerMTok: 2.00, OutputPerMTok: 8.00},
	"gpt-4.1-mini": {InputPerMTok: 0.40, OutputPerMTok: 1.60},
	"gpt-4.1-nano": {InputPerMTok: 0.10, OutputPerMTok: 0.40},
	"o3":           {InputPerMTok: 2.00, OutputPerMTok: 8.00},
	"o3-mini":      {InputPerMTok: 1.10, OutputPerMTok: 4.40},
	"o4-mini":      {InputPerMTok: 1.10, OutputPerMTok: 4.40},

	// ── Google ────────────────────────────────────────────────────
	"gemini-2.5-flash": {InputPerMTok: 0.15, OutputPerMTok: 0.60},
	"gemini-2.5-pro":   {InputPerMTok: 1.25, OutputPerMTok: 10.00},
	"gemini-2.0-flash": {InputPerMTok: 0.10, OutputPerMTok: 0.40},

	// ── Zhipu / Z.ai GLM (public Z.ai API list prices, added May 2026) ─
	"glm-4.6":     {InputPerMTok: 0.60, OutputPerMTok: 2.20},
	"glm-4.5":     {InputPerMTok: 0.60, OutputPerMTok: 2.20},
	"glm-4.5-air": {InputPerMTok: 0.20, OutputPerMTok: 1.10},
}

var priceMu sync.RWMutex
var prices = func() map[string]Price {
	m := make(map[string]Price, len(defaultPrices))
	for k, v := range defaultPrices {
		m[k] = v
	}
	return m
}()

// Lookup returns the price for a model, or (Price{}, false) if unknown.
func Lookup(model string) (Price, bool) {
	priceMu.RLock()
	defer priceMu.RUnlock()
	p, ok := prices[model]
	return p, ok
}

// Register overrides or adds a price. Safe for init() in callers.
func Register(model string, p Price) {
	priceMu.Lock()
	defer priceMu.Unlock()
	prices[model] = p
}

// WarnFunc is invoked, when non-nil, once per distinct model that has no
// price-book entry, the first time a cost is computed for it. Wire it to a
// logger to surface "$0 because the model is unpriced" — otherwise an
// unpriced model is silently accounted as free, which is the failure mode
// that makes a quality-per-dollar benchmark read ∞. Safe to leave nil; the
// pricing package itself takes no logging dependency.
var WarnFunc func(model string)

var unpricedSeen sync.Map // model string -> struct{}; bounds WarnFunc to once per model

// ComputeWithOK returns the USD cost for the given token counts at the
// model's rate and whether the model was actually priced. When the model
// is unknown it returns (0, false) and fires WarnFunc once for that model.
//
// Prefer this over Compute when you report spend: a (0, false) result means
// "cost unknown", which is not the same as a genuinely free (0, true) call.
func ComputeWithOK(model string, in, out int) (float64, bool) {
	p, ok := Lookup(model)
	if !ok {
		if WarnFunc != nil {
			if _, loaded := unpricedSeen.LoadOrStore(model, struct{}{}); !loaded {
				WarnFunc(model)
			}
		}
		return 0, false
	}
	return (float64(in)*p.InputPerMTok + float64(out)*p.OutputPerMTok) / 1_000_000.0, true
}

// Compute returns the USD cost for the given token counts at the model's
// rate, or 0 if the model isn't priced. It is a thin wrapper over
// ComputeWithOK that discards the priced flag; new callers that report
// spend should use ComputeWithOK so unpriced and free are distinguishable.
func Compute(model string, in, out int) float64 {
	cost, _ := ComputeWithOK(model, in, out)
	return cost
}

// KnownModels returns a sorted list of all models with pricing data.
// Useful for diagnostics and documentation.
func KnownModels() []string {
	priceMu.RLock()
	defer priceMu.RUnlock()
	out := make([]string, 0, len(prices))
	for k := range prices {
		out = append(out, k)
	}
	return out
}
