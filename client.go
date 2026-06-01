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
	// RoleTool is a tool-result turn. The message carries the tool's output
	// in Content and links to the originating call via ToolCallID.
	RoleTool Role = "tool"
)

// Message is a single chat turn.
type Message struct {
	Role    Role
	Content string

	// ToolCalls is set on an assistant turn (Role == RoleAssistant) that
	// requested one or more tool invocations. Echo it back in the
	// conversation so the provider sees a well-formed call/response history
	// across a multi-turn tool loop.
	ToolCalls []ToolCall

	// ToolCallID links a tool-result turn (Role == RoleTool) back to the
	// ToolCall.ID it answers. Required on tool-result messages; ignored on
	// all other roles.
	ToolCallID string
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

	// Tools is a provider-agnostic list of tool/function declarations the
	// model may call. When non-empty the adapter forwards them to the
	// provider and populates Response.ToolCalls with any the model invokes.
	Tools []ToolDef

	// ToolChoice steers tool selection. "" lets the provider decide
	// (equivalent to "auto" almost everywhere). "auto" permits zero or more
	// calls; "required" (alias "any") forces at least one call; any other
	// value is treated as a tool name to force. Forcing semantics are
	// best-effort and provider-dependent.
	ToolChoice string
}

// Usage is normalized token + cost accounting for one call.
type Usage struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int

	// CostUSD is the computed price for this call. It is 0 when the model
	// has no price-book entry — inspect Priced to tell that apart from a
	// genuinely zero-cost call.
	CostUSD float64

	// Priced is true when CostUSD came from a known price-book entry. When
	// false, CostUSD is 0 because the model is unpriced (NOT because the
	// call was free); callers reporting spend should treat the value as
	// unknown rather than zero.
	Priced bool
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

	// ToolCalls is the model's request to invoke tools, populated by the
	// adapter when the provider returns tool calls. Empty when the model
	// replied with content only.
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
// (streaming, certain capabilities). Kept as a sentinel so callers can
// branch with errors.Is.
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

// ToolDef is a provider-agnostic tool/function description supplied on a
// Request. InputSchema is the tool's parameter schema as raw JSON Schema
// bytes; the adapter unmarshals it before forwarding to the provider.
type ToolDef struct {
	Name        string
	Description string
	InputSchema []byte // JSON schema
}

// ToolCall is the model's request to invoke a tool, surfaced on a
// Response and echoed back on an assistant Message during a tool loop.
// Input holds the call arguments as raw JSON bytes.
type ToolCall struct {
	ID    string
	Name  string
	Input []byte // JSON
}
