package pricing

// System One models are priced by hand, in this file, because the upstream
// feeds do not carry them.
//
// LiteLLM and OpenRouter both index chat-completion models. A System One
// model is not one — it has no chat endpoint to resell — so it never
// appears in a refreshed snapshot, and an entry added to defaults_gen.go
// would be destroyed the next time `go generate ./pricing` runs.
//
// Hence a separate table, consulted last. Anything a feed or an explicit
// Register does know still wins, so if Jev ever shows up in a snapshot the
// live number takes over without a change here.
//
// Rates are USD per 1,000,000 tokens.
var systemOnePrices = map[string]Price{
	// TypeSafe — read from docs.typesafe.ai/models.md on 2026-09-17.
	//
	// Output tokens are free, and that zero is load-bearing: it is a real
	// published rate, not a gap in the table. Do not "fix" it by copying
	// the input rate across.
	//
	// The versioned ID is the entry that matters. The aliases are listed
	// too so a caller that sends one still prices, but an alias moves when
	// TypeSafe ships a release — price on the ID the response reports, not
	// the one the request asked for.
	"jev-1.13.0":  {InputPerMTok: 0.042, OutputPerMTok: 0},
	"jev-latest":  {InputPerMTok: 0.042, OutputPerMTok: 0},
	"jev-preview": {InputPerMTok: 0.042, OutputPerMTok: 0},
	// The same model as Vercel AI Gateway names it; the gateway adds no
	// markup (vercel.com/ai-gateway/models/jev, read 2026-09-20).
	"typesafe-ai/jev": {InputPerMTok: 0.042, OutputPerMTok: 0},
}

// lookupSystemOne resolves a model against the hand-maintained table.
func lookupSystemOne(model string) (Price, bool) {
	return lookupIn(systemOnePrices, model)
}
