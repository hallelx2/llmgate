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

	// CacheBreakpoint marks this message as the end of a cacheable prefix.
	//
	// Everything from the start of the conversation up to and including
	// this message is stored by the provider and re-read on later calls
	// instead of being reprocessed. That is the single largest cost
	// reduction available to a workload that sends a large, stable prefix
	// on every call — a document, a schema, a long system brief. Anthropic
	// bills a cache read at a tenth of the input rate, so a 120k-token
	// prompt whose first 100k are cached costs roughly a third of the
	// uncached call.
	//
	// Set it on the last message of the stable prefix, not on the part that
	// changes per call: the cache matches on an exact prefix, so a
	// breakpoint after anything variable never hits.
	//
	// Only providers with prompt caching act on this; the rest ignore it.
	// It applies to user messages only — Anthropic's API takes cache
	// markers on content blocks, and a system prompt is not one.
	//
	// Providers meter caching in the usage they return, so Response.Usage
	// reports CacheWrite on the call that populates the cache and CacheRead
	// on the calls that hit it, and cost accounting follows automatically.
	// Caching below a provider's minimum cacheable length (1024 tokens on
	// most Anthropic models) silently does nothing.
	CacheBreakpoint bool
}

// Request is a single completion request.
type Request struct {
	Model     string
	Messages  []Message
	MaxTokens int

	// Temperature is the sampling temperature. nil means "leave it to the
	// provider", which is NOT the same as zero — provider defaults sit
	// around 1.0. Pass Float64(0) for deterministic sampling; a bare 0
	// would be indistinguishable from unset.
	Temperature *float64

	// TopP is nucleus-sampling cutoff. nil leaves it unset. Same
	// zero-is-meaningful reasoning as Temperature.
	TopP *float64

	// Seed requests deterministic sampling where the provider supports
	// it. nil leaves it unset. Best-effort — no provider guarantees it.
	Seed *int

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
//
// The token fields are disjoint: InputTokens counts only uncached prompt
// tokens, so InputTokens + CacheWriteTokens + CacheReadTokens is the full
// prompt size. Providers disagree about this — Anthropic reports cache
// tokens alongside input, OpenAI and Google report them inside it — and
// the adapter normalizes to the disjoint form so a caller can add the
// fields without double-counting.
type Usage struct {
	// InputTokens is prompt tokens billed at the full input rate, i.e.
	// excluding anything served from or written to the prompt cache.
	InputTokens int
	// OutputTokens is all generated tokens, including ReasoningTokens
	// where the provider bills reasoning as output (all of them today).
	OutputTokens int
	// TotalTokens is every token the call touched, cached or not.
	TotalTokens int

	// CacheWriteTokens is prompt tokens written into the provider's cache
	// this call. Billed at a premium — 1.25x input on Anthropic.
	CacheWriteTokens int
	// CacheReadTokens is prompt tokens served from cache. Billed at a
	// discount — 0.1x input on Anthropic, 0.5x on OpenAI, 0.25x on Google.
	CacheReadTokens int
	// ReasoningTokens is the thinking/reasoning portion of OutputTokens.
	// Reported for visibility; it is a subset, not an addition.
	ReasoningTokens int

	// CostUSD is the computed price for this call. It is 0 when the model
	// has no price-book entry — inspect Priced to tell that apart from a
	// genuinely zero-cost call.
	CostUSD float64

	// Priced is true when CostUSD came from a known price-book entry. When
	// false, CostUSD is 0 because the model is unpriced (NOT because the
	// call was free); callers reporting spend should treat the value as
	// unknown rather than zero.
	Priced bool

	// TokensReported is true when the provider actually returned token
	// counts. When false the counts above did not come from the provider,
	// and a CostUSD of 0 with Priced true would otherwise be an assertion
	// that the call was free — which it never is.
	TokensReported bool

	// Estimated is true when the counts were derived from a local
	// tokenizer because the provider reported none. CostUSD is then an
	// approximation, typically within 10-20%, rather than a zero.
	Estimated bool
}

// Response is the model's reply.
type Response struct {
	Content      string
	InputTokens  int // retained for backwards compatibility; mirrors Usage.InputTokens
	OutputTokens int // retained for backwards compatibility; mirrors Usage.OutputTokens
	Model        string
	FinishReason string

	// ReasoningContent is the model's thinking / reasoning trace when the
	// provider exposes it separately from the answer (Claude extended
	// thinking, OpenAI o-series). Empty when the model didn't reason or
	// the provider folds it into Content.
	ReasoningContent string

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

// Ptr returns a pointer to v. It exists so optional Request fields can be
// set inline: Temperature: llmgate.Ptr(0.7).
func Ptr[T any](v T) *T { return &v }

// Float64 returns a pointer to v. Prefer it over Ptr for the sampling
// fields, where the untyped-constant form reads better:
// Temperature: llmgate.Float64(0).
func Float64(v float64) *float64 { return &v }

// Int returns a pointer to v, for Request.Seed.
func Int(v int) *int { return &v }

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
