// Package adapter is the internal seam where llmgate's Client interface
// meets langchaingo's provider implementations. Providers construct an
// Adapter via NewAdapter; callers never see this package directly.
package adapter

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/pkoukk/tiktoken-go"
	"github.com/tmc/langchaingo/llms"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/capabilities"
	"github.com/hallelx2/llmgate/pricing"
)

// Adapter wraps a langchaingo llms.Model and presents it as an
// llmgate.Client. Every Client produced by the provider subpackages is
// an *Adapter under the hood.
type Adapter struct {
	m        llms.Model
	provider llmgate.Provider
	model    string
	modelSet bool // true if model came from config (pass it as a per-call option)
	countTok func(ctx context.Context, text string) (int, error)
}

// NewAdapter constructs an *Adapter for a given langchaingo model.
// provider identifies the vendor for error messages. model is the
// default model name; modelSet is true when the model came from user
// config (so it should be passed as a per-call option).
func NewAdapter(m llms.Model, provider llmgate.Provider, model string, modelSet bool) *Adapter {
	return &Adapter{m: m, provider: provider, model: model, modelSet: modelSet}
}

// SetCountTokens installs an optional token-counting function used by
// CountTokens. Providers may install a tokenizer in their New().
func (a *Adapter) SetCountTokens(f func(ctx context.Context, text string) (int, error)) {
	a.countTok = f
}

// Complete translates a Request into llms.GenerateContent, runs it, and
// maps the ContentResponse back into an llmgate.Response.
func (a *Adapter) Complete(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
	msgs := toLangchainMessages(req.Messages, req.JSONMode, req.JSONSchema)

	opts := []llms.CallOption{}
	if m := req.Model; m != "" {
		opts = append(opts, llms.WithModel(m))
	} else if a.modelSet {
		opts = append(opts, llms.WithModel(a.model))
	}
	if req.MaxTokens > 0 {
		opts = append(opts, llms.WithMaxTokens(req.MaxTokens))
	}
	// Nil, not zero, means "unset" for the sampling knobs — a caller asking
	// for temperature 0 wants determinism, and dropping it silently leaves
	// the request at the provider default (~1.0).
	if req.Temperature != nil {
		opts = append(opts, llms.WithTemperature(*req.Temperature))
	}
	if req.TopP != nil {
		opts = append(opts, llms.WithTopP(*req.TopP))
	}
	if req.Seed != nil {
		opts = append(opts, llms.WithSeed(*req.Seed))
	}
	if len(req.Tools) > 0 {
		opts = append(opts, llms.WithTools(toLangchainTools(req.Tools)))
		if tc := toLangchainToolChoice(req.ToolChoice); tc != nil {
			opts = append(opts, llms.WithToolChoice(tc))
		}
	}

	resp, err := a.m.GenerateContent(ctx, msgs, opts...)
	if err != nil {
		// Try to create a structured LLMError with HTTP status code
		// extracted from the error message.
		if code := extractHTTPStatus(err); code > 0 {
			return nil, llmgate.NewLLMError(
				a.provider, code,
				err.Error(), err,
			)
		}
		return nil, fmt.Errorf("%s: %w", a.provider, err)
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("%s: empty response", a.provider)
	}

	folded := foldChoices(resp.Choices)
	model := req.Model
	if model == "" {
		model = a.model
	}
	out := &llmgate.Response{
		Content:          folded.content,
		ReasoningContent: folded.reasoning,
		Model:            model,
		FinishReason:     folded.finishReason,
		ToolCalls:        folded.toolCalls,
	}

	out.Usage = a.usage(ctx, model, folded, req, out.Content)
	out.InputTokens = out.Usage.InputTokens
	out.OutputTokens = out.Usage.OutputTokens

	return out, nil
}

// usage builds the normalized accounting for one call: it reads whatever
// token breakdown the provider reported, estimates when it reported none,
// and prices the result.
func (a *Adapter) usage(ctx context.Context, model string, f folded, req llmgate.Request, content string) llmgate.Usage {
	tk, reported := extractTokens(f.genInfo)

	u := llmgate.Usage{TokensReported: reported}
	if !reported {
		// The provider told us nothing. Estimating is far better than
		// reporting zero: a zero paired with Priced:true asserts the call
		// was free, which it never is. An estimate is typically within
		// 10-20% and is labelled as such.
		tk = a.estimateTokens(ctx, req, content)
		u.Estimated = true
		warnUnreported(a.provider, model)
	}

	u.InputTokens = tk.Input
	u.OutputTokens = tk.Output
	u.CacheWriteTokens = tk.CacheWrite
	u.CacheReadTokens = tk.CacheRead
	u.ReasoningTokens = tk.Reasoning
	u.TotalTokens = tk.Input + tk.Output + tk.CacheWrite + tk.CacheRead
	u.CostUSD, u.Priced = pricing.ComputeTokens(model, tk)
	return u
}

// extractTokens reads the token breakdown out of a provider's
// GenerationInfo and normalizes it so the fields are disjoint.
//
// The providers disagree about whether cached tokens are counted inside
// the prompt total. Anthropic reports cache_creation and cache_read
// *alongside* input_tokens; OpenAI and Google report cached tokens
// *within* the prompt count. Both are normalized to "Input excludes
// everything cached" so a caller can sum the fields without
// double-counting — and, more importantly, so cached tokens are billed at
// their own rate instead of silently costing nothing.
func extractTokens(gi map[string]any) (pricing.Tokens, bool) {
	if !hasUsage(gi) {
		return pricing.Tokens{}, false
	}

	tk := pricing.Tokens{
		Input:  getInt(gi, "InputTokens", "PromptTokens", "input_tokens", "prompt_tokens"),
		Output: getInt(gi, "OutputTokens", "CompletionTokens", "output_tokens", "completion_tokens"),
		// Anthropic is the only provider that writes to the cache
		// explicitly, and the only one that reports the write separately.
		CacheWrite: getInt(gi, "CacheCreationInputTokens", "cache_creation_input_tokens"),
		CacheRead: getInt(gi, "CacheReadInputTokens", "PromptCachedTokens", "CachedTokens",
			"cache_read_input_tokens", "cached_tokens"),
		Reasoning: getInt(gi, "ReasoningTokens", "ThinkingTokens", "CompletionReasoningTokens", "reasoning_tokens"),
	}

	// Google publishes the already-subtracted figure; prefer it over
	// doing the arithmetic ourselves.
	if n, ok := lookupInt(gi, "NonCachedInputTokens"); ok {
		tk.Input = n
		return tk, true
	}

	// OpenAI and Google fold cache reads into the prompt count, Anthropic
	// does not. Detect which by asking whether subtracting would go
	// negative — that only happens when the counts were already disjoint.
	if tk.CacheRead > 0 && tk.Input >= tk.CacheRead {
		tk.Input -= tk.CacheRead
	}

	return tk, true
}

// estimateTokens approximates a call's token usage from the request and
// the returned text, for providers that report no usage at all.
func (a *Adapter) estimateTokens(ctx context.Context, req llmgate.Request, content string) pricing.Tokens {
	var prompt strings.Builder
	for _, m := range req.Messages {
		prompt.WriteString(m.Content)
		prompt.WriteByte('\n')
		for _, tc := range m.ToolCalls {
			prompt.WriteString(tc.Name)
			prompt.Write(tc.Input)
		}
	}
	for _, t := range req.Tools {
		prompt.WriteString(t.Name)
		prompt.WriteString(t.Description)
		prompt.Write(t.InputSchema)
	}

	in, err := a.CountTokens(ctx, prompt.String())
	if err != nil {
		in = len(prompt.String()) / 4
	}
	out, err := a.CountTokens(ctx, content)
	if err != nil {
		out = len(content) / 4
	}
	return pricing.Tokens{Input: in, Output: out}
}

// UnreportedUsageFunc is invoked, when non-nil, the first time a
// provider/model pair returns a response with no token counts. Wire it to
// a logger: an unreported call is priced from a local estimate, and a
// caller reconciling against an invoice needs to know which calls those
// were.
var UnreportedUsageFunc func(provider llmgate.Provider, model string)

var unreportedSeen sync.Map // provider+model -> struct{}

func warnUnreported(provider llmgate.Provider, model string) {
	if UnreportedUsageFunc == nil {
		return
	}
	if _, loaded := unreportedSeen.LoadOrStore(string(provider)+"/"+model, struct{}{}); !loaded {
		UnreportedUsageFunc(provider, model)
	}
}

// folded is one logical reply assembled from every choice a provider
// returned.
type folded struct {
	content      string
	reasoning    string
	toolCalls    []llmgate.ToolCall
	finishReason string
	genInfo      map[string]any
}

// usageKeys are every key any provider uses to report token counts. A
// choice carrying none of them has no usage to contribute.
var usageKeys = []string{
	"InputTokens", "PromptTokens", "input_tokens", "prompt_tokens",
	"OutputTokens", "CompletionTokens", "output_tokens", "completion_tokens",
}

// foldChoices collapses a provider response into a single reply.
//
// langchaingo's Anthropic adapter returns one ContentChoice per *content
// block*, not per completion candidate — a reply of [thinking, text] or
// [text, tool_use] arrives as two choices. Reading Choices[0] alone
// therefore returned an empty answer whenever the model thought first,
// and silently dropped tool calls whenever it narrated before calling.
//
// Usage is taken from the first choice that reports any, never summed:
// every block carries a copy of the same response-level usage, so adding
// them up would multiply the bill by the block count.
func foldChoices(choices []*llms.ContentChoice) folded {
	var f folded
	var content, reasoning strings.Builder
	seenReasoning := map[string]bool{}

	appendReasoning := func(s string) {
		// Anthropic repeats ThinkingContent on the text block as well as
		// the thinking block; dedupe so it isn't emitted twice.
		if s == "" || seenReasoning[s] {
			return
		}
		seenReasoning[s] = true
		reasoning.WriteString(s)
	}

	for _, c := range choices {
		if c == nil {
			continue
		}
		content.WriteString(c.Content)
		appendReasoning(c.ReasoningContent)
		if s, ok := c.GenerationInfo["ThinkingContent"].(string); ok {
			appendReasoning(s)
		}
		f.toolCalls = append(f.toolCalls, fromLangchainToolCalls(c)...)
		if c.StopReason != "" {
			f.finishReason = c.StopReason
		}
		if f.genInfo == nil && hasUsage(c.GenerationInfo) {
			f.genInfo = c.GenerationInfo
		}
	}

	// No choice reported usage — keep the first non-nil map so any other
	// metadata is still reachable downstream.
	if f.genInfo == nil {
		for _, c := range choices {
			if c != nil && c.GenerationInfo != nil {
				f.genInfo = c.GenerationInfo
				break
			}
		}
	}

	f.content = content.String()
	f.reasoning = reasoning.String()
	return f
}

// hasUsage reports whether m carries any recognised token-count key.
func hasUsage(m map[string]any) bool {
	if m == nil {
		return false
	}
	for _, k := range usageKeys {
		if _, ok := m[k]; ok {
			return true
		}
	}
	return false
}

// toLangchainTools converts our provider-agnostic tool declarations into
// langchaingo's []llms.Tool. The InputSchema bytes are unmarshalled into
// the generic structure langchaingo forwards as the function parameters;
// a malformed or empty schema yields a tool with no declared parameters.
func toLangchainTools(defs []llmgate.ToolDef) []llms.Tool {
	out := make([]llms.Tool, 0, len(defs))
	for _, d := range defs {
		fn := &llms.FunctionDefinition{
			Name:        d.Name,
			Description: d.Description,
		}
		if len(d.InputSchema) > 0 {
			var params any
			if err := json.Unmarshal(d.InputSchema, &params); err == nil {
				fn.Parameters = params
			}
		}
		out = append(out, llms.Tool{Type: "function", Function: fn})
	}
	return out
}

// toLangchainToolChoice maps the provider-agnostic ToolChoice string onto
// langchaingo's WithToolChoice argument. Returns nil to leave the choice
// unset (provider default ~ "auto").
func toLangchainToolChoice(choice string) any {
	switch c := strings.ToLower(strings.TrimSpace(choice)); c {
	case "":
		return nil
	case "auto", "none", "required":
		return c // strings every provider understands
	case "any":
		return "required" // normalise Anthropic's spelling to the common one
	default:
		// Treat anything else as a request to force a specific named tool.
		return llms.ToolChoice{
			Type:     "function",
			Function: &llms.FunctionReference{Name: choice},
		}
	}
}

// fromLangchainToolCalls maps a response choice's tool calls into our
// ToolCall slice. It prefers the plural ToolCalls field and falls back to
// the singular FuncCall that some providers populate instead.
func fromLangchainToolCalls(choice *llms.ContentChoice) []llmgate.ToolCall {
	if choice == nil {
		return nil
	}
	raw := choice.ToolCalls
	if len(raw) == 0 && choice.FuncCall != nil {
		raw = []llms.ToolCall{{Type: "function", FunctionCall: choice.FuncCall}}
	}
	if len(raw) == 0 {
		return nil
	}
	out := make([]llmgate.ToolCall, 0, len(raw))
	for _, tc := range raw {
		call := llmgate.ToolCall{ID: tc.ID}
		if tc.FunctionCall != nil {
			call.Name = tc.FunctionCall.Name
			if tc.FunctionCall.Arguments != "" {
				call.Input = []byte(tc.FunctionCall.Arguments)
			}
		}
		out = append(out, call)
	}
	return out
}

// Capabilities reports known capabilities for the adapter's configured
// model by looking them up in the registry. Satisfies capabilities.Capable.
func (a *Adapter) Capabilities() capabilities.Capabilities {
	return capabilities.Lookup(a.model)
}

// CountTokens uses tiktoken-go for accurate token counting when
// available, falling back to a per-provider tokenizer or the ~4 chars
// per token heuristic.
func (a *Adapter) CountTokens(ctx context.Context, text string) (int, error) {
	// Priority 1: per-provider override (installed by factory).
	if a.countTok != nil {
		return a.countTok(ctx, text)
	}

	// Priority 2: tiktoken-go encoding lookup.
	// tiktoken supports OpenAI models (cl100k_base, o200k_base) and
	// falls back to cl100k_base for unknown models. This is
	// significantly more accurate than len/4 for all providers.
	enc, err := tiktoken.EncodingForModel(a.model)
	if err != nil {
		// Model not in tiktoken's registry — try common base encodings.
		enc, err = tiktoken.GetEncoding("cl100k_base")
	}
	if err == nil {
		tokens := enc.Encode(text, nil, nil)
		return len(tokens), nil
	}

	// Priority 3: rough heuristic (~4 chars per token).
	return len(text) / 4, nil
}

// toLangchainMessages translates our Message slice into llms.MessageContent.
// When JSONMode is on, appends a firm "reply with JSON only" nudge to the
// last human message — providers differ on strict JSON mode support, so the
// prompt nudge is the one approach that works everywhere.
func toLangchainMessages(msgs []llmgate.Message, jsonMode bool, schema []byte) []llms.MessageContent {
	out := make([]llms.MessageContent, 0, len(msgs))
	for _, m := range msgs {
		role := toLangchainRole(m.Role)
		var parts []llms.ContentPart

		switch m.Role {
		case llmgate.RoleTool:
			// Tool-result turn: carry the output keyed by the call ID so the
			// provider can match it to the assistant's request.
			parts = append(parts, llms.ToolCallResponse{
				ToolCallID: m.ToolCallID,
				Content:    m.Content,
			})
		default:
			// Keep the text part for plain turns and for assistant turns that
			// pair a message with tool calls. Skip the empty placeholder only
			// when the turn is purely tool calls.
			if m.Content != "" || len(m.ToolCalls) == 0 {
				parts = append(parts, llms.TextContent{Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				parts = append(parts, llms.ToolCall{
					ID:   tc.ID,
					Type: "function",
					FunctionCall: &llms.FunctionCall{
						Name:      tc.Name,
						Arguments: string(tc.Input),
					},
				})
			}
		}

		out = append(out, llms.MessageContent{Role: role, Parts: parts})
	}

	if !jsonMode {
		return out
	}

	nudge := "\n\nRespond with ONLY a single JSON object. No prose, no code fences."
	if len(schema) > 0 {
		nudge += " The object must conform to this JSON schema:\n" + string(schema)
	}
	// Append the nudge to the last human message, or add a new one.
	for i := len(out) - 1; i >= 0; i-- {
		if out[i].Role == llms.ChatMessageTypeHuman {
			if n := len(out[i].Parts); n > 0 {
				if tc, ok := out[i].Parts[n-1].(llms.TextContent); ok {
					out[i].Parts[n-1] = llms.TextContent{Text: tc.Text + nudge}
					return out
				}
			}
		}
	}
	out = append(out, llms.TextParts(llms.ChatMessageTypeHuman, nudge))
	return out
}

func toLangchainRole(r llmgate.Role) llms.ChatMessageType {
	switch r {
	case llmgate.RoleSystem:
		return llms.ChatMessageTypeSystem
	case llmgate.RoleAssistant:
		return llms.ChatMessageTypeAI
	case llmgate.RoleTool:
		return llms.ChatMessageTypeTool
	default:
		return llms.ChatMessageTypeHuman
	}
}

// getInt pulls the first present integer under the given keys out of a
// map[string]any, or 0 when none are present.
func getInt(m map[string]any, keys ...string) int {
	n, _ := lookupInt(m, keys...)
	return n
}

// lookupInt is getInt with presence reported separately, so a genuine 0
// can be told apart from a missing key. Copes with the several numeric
// types providers decode into.
func lookupInt(m map[string]any, keys ...string) (int, bool) {
	if m == nil {
		return 0, false
	}
	for _, k := range keys {
		v, ok := m[k]
		if !ok {
			continue
		}
		switch x := v.(type) {
		case int:
			return x, true
		case int32:
			return int(x), true
		case int64:
			return int(x), true
		case uint32:
			return int(x), true
		case uint64:
			return int(x), true
		case float64:
			return int(x), true
		}
	}
	return 0, false
}

// httpStatusRe matches "status code: NNN" or "status NNN" or just a
// bare 3-digit HTTP code preceded by a word boundary.
var httpStatusRe = regexp.MustCompile(`(?i)(?:status[ _]?(?:code)?:?\s*)(\d{3})`)

// extractHTTPStatus tries to pull an HTTP status code from an error
// message. Returns 0 if none found.
func extractHTTPStatus(err error) int {
	if err == nil {
		return 0
	}
	msg := err.Error()
	// Fast path: look for common patterns.
	if m := httpStatusRe.FindStringSubmatch(msg); len(m) > 1 {
		if code, e := strconv.Atoi(m[1]); e == nil && code >= 100 && code < 600 {
			return code
		}
	}
	// Check for bare status codes in known error format "API returned
	// unexpected status code: NNN"
	for _, prefix := range []string{
		"unexpected status code: ",
		"status code: ",
	} {
		idx := strings.Index(strings.ToLower(msg), prefix)
		if idx < 0 {
			continue
		}
		sub := msg[idx+len(prefix):]
		if len(sub) >= 3 {
			if code, e := strconv.Atoi(sub[:3]); e == nil && code >= 100 && code < 600 {
				return code
			}
		}
	}
	return 0
}
