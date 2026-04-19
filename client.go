package llmgate

import (
	"context"
	"errors"
)

// Role identifies the speaker of a message.
type Role string

const (
	// RoleSystem is the system / developer instructions role.
	RoleSystem Role = "system"
	// RoleUser is the end-user / human role.
	RoleUser Role = "user"
	// RoleAssistant is the model's own prior-turn role.
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

	// Tools is a provider-agnostic list of tool/function declarations.
	// Scaffolding only — not wired into the adapter yet.
	Tools []ToolDef
}

// Usage is normalized token + cost accounting for one call.
type Usage struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int
	CostUSD      float64 // 0 if price unknown for the model
}

// Response is the model's reply.
type Response struct {
	Content      string
	InputTokens  int // retained for backwards compatibility; mirrors Usage.InputTokens
	OutputTokens int // retained for backwards compatibility; mirrors Usage.OutputTokens
	Model        string
	FinishReason string

	// Usage is the normalized accounting for this call.
	Usage Usage

	// FromCache is true when the response was served by the cache middleware
	// without invoking the underlying provider.
	FromCache bool

	// ToolCalls is the model's request to invoke tools. Scaffolding only —
	// not populated by the adapter yet.
	ToolCalls []ToolCall
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

// Middleware wraps a Client. Compose them: retry.New(...)(cache.New(...)(base)).
type Middleware func(Client) Client

// Provider identifies an LLM vendor.
type Provider string

const (
	// ProviderAnthropic identifies Anthropic Claude.
	ProviderAnthropic Provider = "anthropic"
	// ProviderOpenAI identifies OpenAI.
	ProviderOpenAI Provider = "openai"
	// ProviderGemini identifies Google Gemini.
	ProviderGemini Provider = "gemini"
)

// ErrNotImplemented is returned by surfaces that aren't wired up yet
// (streaming, tool use, certain capabilities). Kept as a sentinel so
// callers can branch with errors.Is.
var ErrNotImplemented = errors.New("llmgate: not yet implemented")

// StreamChunk is one event in a streaming completion.
type StreamChunk struct {
	Delta        string
	FinishReason string
	Usage        *Usage // only set on the terminal chunk, may be nil
}

// Streamer is an optional extension a Client may implement for streaming.
// Callers type-assert: if s, ok := client.(Streamer); ok { ... }.
//
// Concrete provider implementations are pending — the adapter does not
// yet implement Streamer.
type Streamer interface {
	Stream(ctx context.Context, req Request) (<-chan StreamChunk, error)
}

// ToolDef is a provider-agnostic tool/function description.
//
// Scaffolding: declared on Request but not yet plumbed through the adapter.
type ToolDef struct {
	Name        string
	Description string
	InputSchema []byte // JSON schema
}

// ToolCall is the model's request to invoke a tool.
//
// Scaffolding: declared on Response but not yet populated by the adapter.
type ToolCall struct {
	ID    string
	Name  string
	Input []byte // JSON
}
