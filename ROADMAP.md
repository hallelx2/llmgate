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
| Phase 3 — streaming + tool use | Deferred |
| Phase 4 — independent release cadence (CI, release workflow, pkg.go.dev) | Shipped |

Phases 0–2 and 4 cover everything `vectorless-engine` exercises today.
Phase 3 is deferred — the interface types are declared, no caller
needs the behaviour yet, and we'll wire it when one does.

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

### Tool use / function calling

- Types (`Tool`, `ToolCall`, `ToolResult`) declared on `Request` /
  `Response`; adapter currently ignores them.
- Langchaingo has `llms.Tool` and per-provider call/response plumbing.
  The internal adapter needs a two-way translation: llmgate tools →
  `llms.Tool` on the way in, `llms.ContentChoice.ToolCalls` →
  `Response.ToolCalls` on the way out.
- Each provider subpackage grows a capability flag if the backend
  doesn't support tools (some Gemini models don't).

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
- Pricing table drift — prices change. A future job could scrape
  vendor pricing pages weekly and open a PR against `pricing/`.
