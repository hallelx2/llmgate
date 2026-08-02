# llmgate roadmap

Design doc and long-form context still live in the engine repo:

- [vectorless-engine/docs/roadmaps/LLMGATE.md](https://github.com/hallelx2/vectorless-engine/blob/main/docs/roadmaps/LLMGATE.md)
- [vectorless-engine/docs/LLMGATE.md](https://github.com/hallelx2/vectorless-engine/blob/main/docs/LLMGATE.md) — design doc

## Current status

| Phase | Status |
|---|---|
| Phase 0 — foundation (interface, providers, mock) | Shipped |
| Phase 1 — swap to langchaingo | Shipped |
| Phase 2 — router / cost / capabilities / middleware (retry, budget, cache) | Shipped |
| Phase 3a — tool use | Shipped |
| Phase 3b — streaming | Deferred |
| Phase 4 — independent release cadence (CI, release workflow, pkg.go.dev) | Shipped |

Tool calling shipped: the adapter translates `Request.Tools` into
provider tool declarations and maps tool calls back onto
`Response.ToolCalls`, across all three providers.

Streaming is still deferred — the interface types are declared and no
provider implements them.

## Deferred — pick up when a caller needs it

These are intentionally not in the critical path. Listed here so we
don't forget the shape of the work.

### Streaming (`Client.Stream`)

- Interface + chunk types declared; no provider implements it yet.
- Langchaingo exposes streaming via `llms.WithStreamingFunc`; the
  internal adapter needs a `Stream` method that adapts that callback
  into an `iter.Seq2[Chunk, error]` (or a channel — decide at impl
  time).
- Middleware implications: `cache.New` should replay cached responses
  as a single chunk; `retry.New` should only retry pre-first-chunk
  failures; `budget.New` debits on stream completion.

### Native `CountTokens` per provider

- Today `CountTokens` is a heuristic estimate in the adapter.
- Anthropic has a `/v1/messages/count_tokens` endpoint; OpenAI has
  `tiktoken` (pure Go ports exist); Gemini has `CountTokens` on
  `GenerativeModel`.
- Replace the estimate per-provider; keep the estimate as the
  fallback so callers never see a hard error from token counting.

### Provider-specific features currently flagged but not wired

- **Anthropic prompt caching** — `Config.EnablePromptCache` exists,
  langchaingo's anthropic adapter doesn't expose `cache_control` yet.
  Either upstream the feature to langchaingo or drop to raw HTTP for
  this one call.
- **OpenAI structured outputs** — `Request.ResponseFormat` could carry
  a JSON schema; langchaingo supports `response_format` but not
  `strict: true` yet.
- **Reasoning models** — `Config.ReasoningModel` is carried on every
  provider Config and unused. Intent was a second Complete path
  (`CompleteReason`? or a Request flag) that routes to o3 / claude
  opus-reason / gemini-pro thinking. Decide the interface shape when
  we actually need it; don't pre-commit.

### Minor

- Per-provider default capability sets in `capabilities/` — today the
  defaults are a one-size-fits-all guess. Split per provider-model
  when a caller relies on the flags for routing.
- Pricing table drift — addressed by `pricing.UseRemote`, which layers
  a refreshed snapshot from LiteLLM or OpenRouter over the embedded
  table. Still open: a scheduled job to regenerate the embedded
  defaults so the offline path does not age either.
