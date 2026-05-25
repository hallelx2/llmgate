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
	"claude-sonnet-4-5":   {InputPerMTok: 3.00, OutputPerMTok: 15.00},
	"claude-sonnet-4-20250514": {InputPerMTok: 3.00, OutputPerMTok: 15.00},
	"claude-opus-4-1":     {InputPerMTok: 15.00, OutputPerMTok: 75.00},
	"claude-haiku-4-5":    {InputPerMTok: 1.00, OutputPerMTok: 5.00},
	"claude-haiku-3-5":    {InputPerMTok: 0.80, OutputPerMTok: 4.00},

	// ── OpenAI ────────────────────────────────────────────────────
	"gpt-4o":              {InputPerMTok: 2.50, OutputPerMTok: 10.00},
	"gpt-4o-mini":         {InputPerMTok: 0.15, OutputPerMTok: 0.60},
	"gpt-4.1":             {InputPerMTok: 2.00, OutputPerMTok: 8.00},
	"gpt-4.1-mini":        {InputPerMTok: 0.40, OutputPerMTok: 1.60},
	"gpt-4.1-nano":        {InputPerMTok: 0.10, OutputPerMTok: 0.40},
	"o3":                  {InputPerMTok: 2.00, OutputPerMTok: 8.00},
	"o3-mini":             {InputPerMTok: 1.10, OutputPerMTok: 4.40},
	"o4-mini":             {InputPerMTok: 1.10, OutputPerMTok: 4.40},

	// ── Google ────────────────────────────────────────────────────
	"gemini-2.5-flash":    {InputPerMTok: 0.15, OutputPerMTok: 0.60},
	"gemini-2.5-pro":      {InputPerMTok: 1.25, OutputPerMTok: 10.00},
	"gemini-2.0-flash":    {InputPerMTok: 0.10, OutputPerMTok: 0.40},
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

// Compute returns the USD cost for the given token counts at the
// model's rate, or 0 if the model isn't priced.
func Compute(model string, in, out int) float64 {
	p, ok := Lookup(model)
	if !ok {
		return 0
	}
	return (float64(in)*p.InputPerMTok + float64(out)*p.OutputPerMTok) / 1_000_000.0
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
