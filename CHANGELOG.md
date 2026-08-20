# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0, a breaking change bumps the **minor** version.

This file starts at 0.4.0. Earlier releases are described by their
[GitHub Releases](https://github.com/hallelx2/llmgate/releases).

## [Unreleased]

Nothing yet.

## [0.4.0] - 2026-08-20

The reported cost of a call was wrong in four independent ways, and the
flag for the single largest available saving did nothing. Both are fixed
here.

### Breaking

`Usage` and `Price` both gained fields. Neither is a rename or a removal,
so callers that use field names compile unchanged — but **positional
struct literals and exhaustive comparisons break**.

`Usage` (`client.go`) gained five fields:

```go
// before (v0.3.0)
type Usage struct {
    InputTokens  int
    OutputTokens int
    TotalTokens  int
    CostUSD      float64
    Priced       bool
}

// after (v0.4.0)
type Usage struct {
    InputTokens      int
    OutputTokens     int
    TotalTokens      int
    CacheWriteTokens int   // new
    CacheReadTokens  int   // new
    ReasoningTokens  int   // new
    CostUSD          float64
    Priced           bool
    TokensReported   bool  // new
    Estimated        bool  // new
}
```

`Price` (`pricing`) gained three:

```go
// before (v0.3.0)
type Price struct {
    InputPerMTok  float64
    OutputPerMTok float64
}

// after (v0.4.0)
type Price struct {
    InputPerMTok      float64
    OutputPerMTok     float64
    CacheWritePerMTok float64  // new; zero falls back to input x family multiplier
    CacheReadPerMTok  float64  // new; zero falls back to input x family multiplier
    ReasoningPerMTok  float64  // new; zero bills at output, which is what every
                               // provider does today
}
```

Migration: name your fields. `Usage{InputTokens: n, OutputTokens: m}`
compiles across both versions; `Usage{n, m, t, c, p}` does not.

`pricing.Compute` and `pricing.ComputeWithOK` are **deprecated** in
favour of `ComputeTokens`, which takes a `Tokens` breakdown. Both still
work and stay behaviour-compatible for callers with no cached tokens, so
this is a deprecation rather than a removal.

### Added

- **Prompt caching is wired.** `Config.EnablePromptCache` had been a
  struct field that did nothing since Phase 0, documented as waiting on
  langchaingo to expose `cache_control`. That constraint lapsed —
  v0.1.14, already in `go.mod`, ships `llms.WithCacheControl` — and the
  comment outlasted it, taking the biggest saving available with it.
  Measured against live rates on a 120k-token prompt whose first 100k are
  a stable prefix: **glm-4.6 falls 66%, claude-sonnet-4-5 73%, gpt-4o
  44%** per warm hop. That is the shape of every vectorless tree
  navigation — a large document resent on each call, followed by a short
  question.

  `Message.CacheBreakpoint` marks the end of a cacheable prefix, and
  `Config.EnablePromptCache` is shorthand for the common shape (cache the
  first user message). An explicit breakpoint anywhere leaves the request
  untouched, since adding a second would cache a prefix nobody chose and
  providers cap how many a request may carry.

  Two constraints are enforced rather than documented and hoped for. The
  marker only goes on **user** turns: langchaingo's system handler takes
  a bare `TextContent` and returns `ErrInvalidContentType` for anything
  else, so marking a system message would not cache it, it would fail
  every request. Assistant and tool turns are excluded for a different
  reason — a breakpoint belongs at the end of a *stable* prefix, and
  those turns are part of what varies.

  The message slice is copied rather than marked in place, because
  `Request.Messages` belongs to the caller and a retry may hand the same
  backing array back.
- `Usage.CacheWriteTokens`, `Usage.CacheReadTokens` and
  `Usage.ReasoningTokens`, normalized to a disjoint form so the fields
  can be summed on any provider regardless of whether that provider
  reports cached tokens inside or outside the prompt count.
- `Usage.TokensReported` and `Usage.Estimated`, so a caller can tell
  provider-reported counts from a local-tokenizer estimate.
- `pricing.UseRemote` layers a refreshed snapshot from LiteLLM or
  OpenRouter over the embedded defaults, under any `Register` overrides.
  Opt-in — importing the package does no network I/O — and fails open
  throughout: lookups never touch the network, a failed fetch keeps the
  last snapshot, a snapshot is persisted so restarts start warm, and one
  whose rates have moved more than 10x from the embedded values is
  rejected as a units error rather than adopted.
- `pricing/gen` generates the embedded price table from the live feed, so
  the offline defaults stop being hand-maintained.

### Fixed

- fix(pricing): **cache and reasoning tokens were dropped entirely.**
  langchaingo already surfaced `CacheCreationInputTokens`,
  `CacheReadInputTokens`, `PromptCachedTokens` and `ReasoningTokens`;
  llmgate read none of them. Anthropic reports cache tokens *outside*
  `input_tokens`, so a cached prompt — writes bill at 1.25x input — was
  charged **nothing at all**. OpenAI and Google fold cached tokens *into*
  the prompt count, so those were charged at the **full rate** instead of
  the 0.5x / 0.25x they actually cost.
- fix(pricing): **lookup was an exact map hit**, so every ID a real API
  returns missed it and priced at **$0** — dated snapshots
  (`claude-sonnet-4-5-20250929`), SDK prefixes
  (`models/gemini-2.5-flash`), Bedrock ARNs, and the gateway-qualified
  forms the aggregate feeds use for the model we actually run
  (`z-ai/glm-4.6`). `Canonical` strips prefixes, dates and revisions,
  then falls back to a longest-prefix match — longest, because
  `claude-sonnet-4` also prefixes `claude-sonnet-4-5-preview` and the
  shorter key would misprice it. Rates are keyed by model ID alone, never
  provider+model: vectorless runs GLM-4.6 through z.ai's
  Anthropic-compatible gateway, and a provider-scoped key would look up
  `anthropic/glm-4.6` and miss.
- fix(pricing): **missing usage was reported as a priced $0 call.** When
  `GenerationInfo` carried no counts, `ComputeWithOK(model, 0, 0)`
  returned `(0, true)` — `Priced: true` actively asserting that a call
  which returned content was free. The adapter now estimates from the
  tokenizer and flags it, rather than reporting a confident zero.
- fix(pricing): **the embedded table had drifted silently** —
  gemini-2.5-flash output billed at 0.24x actual, gpt-4o at 0.91x.
- fix(pricing): price collisions resolve by **provider rank, not map
  iteration order**, so the same model no longer bills at three different
  rates across process starts.
- fix(adapter): **`Temperature: 0` was silently discarded**, so every
  vectorless call ran at the provider default instead of deterministic.
- fix(adapter): the adapter read `Choices[0]` only, so **multi-block
  Claude and GLM responses lost tool calls and returned empty content**.
- fix(middleware): `budget.New` returned one shared client and rewrote
  its inner pointer, so **two wrapped clients silently became one**.
- fix(middleware): the cache key omitted `ToolChoice` and message tool
  calls, producing **wrong cache hits inside tool loops**.
- fix(middleware): `retry` now honours `Retry-After`, and `sleepBackoff`
  no longer panics on overflow or on a sub-2ns `BaseDelay`.
- fix(middleware): OpenAI- and Anthropic-compatible gateways that answer
  **HTTP 200 with an error body** were classified `Unknown` and retried
  four times. They are now recognised as the errors they are.
- fix(cache): JSON mode appended its nudge by type-asserting the last
  part to `TextContent`. A cached part is not one, so the nudge would
  have been dropped silently and **JSON mode would have become a no-op
  whenever caching was on**. It now lands as its own part, outside the
  cached block — also where it belongs, since the nudge carries a
  per-call schema and changing the tail of a cached prefix invalidates it
  on every request.

### Docs

- `ROADMAP.md` no longer lists prompt caching as blocked on langchaingo,
  and no longer implies the price-table generator is unwritten.
- `CONTRIBUTING.md` records the rule this release was nearly mis-numbered
  for: **squash-merge discards per-commit subjects, so a breaking change
  must carry its `!` in the PR title.**

[Unreleased]: https://github.com/hallelx2/llmgate/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/hallelx2/llmgate/releases/tag/v0.4.0
