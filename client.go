// Package llmgate is a provider-agnostic LLM gateway for Go.
//
// It defines a small Client interface that every supported provider
// implements, plus request/response types that cover the subset of
// features the gateway cares about (chat completion, token counting,
// structured output, and — later — streaming, tool use, and
// capability introspection).
//
// Ships with:
//   - A live Anthropic client (direct HTTP, retries, count_tokens).
//   - OpenAI and Gemini stubs returning ErrNotImplemented.
//   - A Mock client for tests.
//
// Forthcoming (see ROADMAP.md):
//   - langchaingo-backed adapters for every provider.
//   - Router with per-provider fallback.
//   - Cost tracking, capability flags, retry/cache/budget middleware.
package llmgate

import (
	"context"
	"errors"
)

// Role identifies the speaker of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is a single chat turn.
type Message struct {
	Role    Role
	Content string
}

// Request is a single completion request.
type Request struct {
	Model       string
	Messages    []Message
	MaxTokens   int
	Temperature float64

	// JSONMode asks the provider to return a JSON object that conforms to
	// JSONSchema. Providers that don't support structured outputs natively
	// should fall back to prompt instruction.
	JSONMode   bool
	JSONSchema []byte
}

// Response is the model's reply.
type Response struct {
	Content      string
	InputTokens  int
	OutputTokens int
	Model        string
	FinishReason string
}

// Client is the provider-agnostic contract.
type Client interface {
	// Complete runs a single completion.
	Complete(ctx context.Context, req Request) (*Response, error)

	// CountTokens returns an approximate token count for text under this
	// client's model. Implementations may use a local tokenizer or the
	// provider's counting endpoint.
	CountTokens(ctx context.Context, text string) (int, error)
}

// Provider identifies an LLM vendor.
type Provider string

const (
	ProviderAnthropic Provider = "anthropic"
	ProviderOpenAI    Provider = "openai"
	ProviderGemini    Provider = "gemini"
)

// ErrNotImplemented is returned by stub providers during scaffolding.
var ErrNotImplemented = errors.New("llmgate: provider not yet implemented")
