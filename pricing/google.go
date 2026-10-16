package pricing

// Gemini models newer than the compiled-in feed snapshot, priced by hand
// from Google's own list.
//
// defaults_gen.go was generated on 2026-08-02 and stops at Gemini 2.5. Two
// things go wrong for a newer ID without this table: Gemini 3 models do not
// price at all, and "gemini-2.5-flash-lite" resolves by longest prefix to
// "gemini-2.5-flash" and is billed at three times its real input rate.
// So this table is consulted before the compiled-in defaults (after
// explicit overrides and any remote snapshot, which stay authoritative).
//
// Source: cloud.google.com/vertex-ai/generative-ai/pricing, read
// 2026-10-16. Rates are the global-endpoint, ≤200K-context, standard
// (on-demand) list prices in USD per 1,000,000 tokens; thinking bills as
// output. Regional endpoints list Gemini 3 at 10% more; the vertex
// provider applies that uplift itself.
//
// Gemini 3.6/3.7/3.8 Flash are listed at $1.50/$7.50 with "introductory
// pricing of $0.75/$3.75 through December 31, 2026", delivered as 50%
// credits back on net spend after the month closes. The invoice line is
// the list price, so that is what is recorded here; the credit-back is a
// later adjustment, not a lower rate.
var googlePrices = map[string]Price{
	"gemini-2.5-flash-lite": {InputPerMTok: 0.10, OutputPerMTok: 0.40, CacheReadPerMTok: 0.01},
	"gemini-3.1-flash-lite": {InputPerMTok: 0.25, OutputPerMTok: 1.50, CacheReadPerMTok: 0.025},
	"gemini-3.5-flash-lite": {InputPerMTok: 0.30, OutputPerMTok: 2.50, CacheReadPerMTok: 0.03},
	"gemini-3.5-flash":      {InputPerMTok: 1.50, OutputPerMTok: 9.00, CacheReadPerMTok: 0.15},
	"gemini-3.6-flash":      {InputPerMTok: 1.50, OutputPerMTok: 7.50, CacheReadPerMTok: 0.15},
	"gemini-3.7-flash":      {InputPerMTok: 1.50, OutputPerMTok: 7.50, CacheReadPerMTok: 0.15},
	"gemini-3.8-flash":      {InputPerMTok: 1.50, OutputPerMTok: 7.50, CacheReadPerMTok: 0.15},
	"gemini-3.1-pro":        {InputPerMTok: 2.00, OutputPerMTok: 12.00, CacheReadPerMTok: 0.20},
}

// lookupGoogle resolves a model against the hand-maintained Gemini table.
func lookupGoogle(model string) (Price, bool) {
	return lookupIn(googlePrices, model)
}
