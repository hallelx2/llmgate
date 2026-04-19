// Package llmgate is a provider-agnostic LLM gateway for Go, built on
// top of github.com/tmc/langchaingo.
//
// langchaingo handles the per-provider wire protocols; llmgate wraps
// them behind one small Client interface and adds the production
// concerns langchaingo doesn't ship — retries, router/fallback, cost
// tracking, capability flags, budget + cache middleware, and error
// classification.
//
// The root package declares only the shared contract: Client, Request,
// Response, Usage, Message, Role, Provider, Streamer, Tool types,
// Middleware, ErrNotImplemented, and the Mock client. Concrete provider
// factories, middleware, router, pricing, and capability helpers live in
// subpackages:
//
//   - github.com/hallelx2/llmgate/provider/anthropic
//   - github.com/hallelx2/llmgate/provider/openai
//   - github.com/hallelx2/llmgate/provider/gemini
//   - github.com/hallelx2/llmgate/middleware/retry
//   - github.com/hallelx2/llmgate/middleware/budget
//   - github.com/hallelx2/llmgate/middleware/cache
//   - github.com/hallelx2/llmgate/router
//   - github.com/hallelx2/llmgate/pricing
//   - github.com/hallelx2/llmgate/capabilities
//
// Forthcoming:
//   - Streaming + tool-use concrete provider implementations.
//   - Native count_tokens via each provider's counting endpoint.
package llmgate
