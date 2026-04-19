# llmgate

> LiteLLM for Go. A single provider-agnostic client over Anthropic,
> OpenAI, Gemini, and the rest — with a router, fallback, cost
> tracking, and capability flags on top.

`llmgate` is the LLM gateway that [vectorless-engine](https://github.com/hallelx2/vectorless-engine)
depends on, extracted into its own module so anything written in Go
can use it. It is **not** a rewrite of LiteLLM; it sits on top of
[`tmc/langchaingo`](https://github.com/tmc/langchaingo) as the
provider-adapter layer (eventually — today we ship a direct
Anthropic HTTP client and stubs for the rest) and adds the
production features langchaingo deliberately doesn't include.

## Why this exists

The Go ecosystem has two extremes:

- **Vendor SDKs** (`openai-go`, `anthropic-sdk-go`) — great typing,
  no portability. Swap providers, rewrite your call site.
- **Thin wrappers** — portable, but you lose cost, retries,
  fallbacks, and capability introspection.

`llmgate` is the middle layer. One interface. Every provider behind
it. All the production concerns — router, fallback on rate-limit,
cost per call, capability flags — baked in rather than bolted on.

## Status

This is early code. It currently lives alongside `vectorless-engine`
and will track its roadmap (see
[vectorless-engine/docs/roadmaps/LLMGATE.md](https://github.com/hallelx2/vectorless-engine/blob/main/docs/roadmaps/LLMGATE.md)).
The interface is the stable surface — implementations will change
underneath as we swap the Anthropic HTTP client for a langchaingo
adapter and fill out the other providers.

**What works today:**

- `Client` interface with `Complete` + `CountTokens`
- Anthropic live client (direct HTTP, retries, `count_tokens`)
- OpenAI and Gemini stubs returning `ErrNotImplemented`
- Mock client for tests

**Coming next (Phase 1+):**

- langchaingo adapter so OpenAI, Gemini, Bedrock, Ollama all work
- Router with per-provider fallback (`OnRateLimit`, `OnStatus`, …)
- Cost tracking via a static price table + `Usage.CostUSD`
- Capability flags (`MaxContext`, `SupportsJSONMode`, `SupportsStreaming`, …)
- Middleware: retries with jitter, in-memory LRU cache, budget guardrails
- Streaming + tool-use

## Install

```bash
go get github.com/hallelx2/llmgate
```

## Use

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/hallelx2/llmgate"
)

func main() {
    client := llmgate.NewAnthropic(llmgate.AnthropicConfig{
        APIKey: os.Getenv("ANTHROPIC_API_KEY"),
        Model:  "claude-sonnet-4-5",
    })

    resp, err := client.Complete(context.Background(), llmgate.Request{
        Messages: []llmgate.Message{
            {Role: llmgate.RoleUser, Content: "In one sentence: what is vectorless retrieval?"},
        },
        MaxTokens: 256,
    })
    if err != nil {
        panic(err)
    }

    fmt.Println(resp.Content)
}
```

## Design principles

- **The interface is tiny.** `Complete`, `CountTokens`, later
  `Stream`, later `Capabilities`. If it doesn't fit, it goes in a
  middleware, not the interface.
- **Middleware over inheritance.** Retries, caching, cost tracking,
  rate-limiting — all `func(Client) Client` wrappers. Compose them.
- **No magic config.** No viper, no auto-reload, no remote backends.
  Pass a struct, get a client.
- **Provider-specific features are honoured where they matter.**
  Anthropic prompt caching, OpenAI structured outputs, Gemini long
  context — opt in via config, not via a lowest-common-denominator
  API.
- **Pure IO-bound code.** Parallelism is always network-bound.
  `errgroup` + `semaphore`, no worker pools.

See [DESIGN.md](./DESIGN.md) for the longer-form write-up and
[ROADMAP.md](./ROADMAP.md) for phases.

## License

Apache 2.0. See [LICENSE](./LICENSE).
