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
	if req.Temperature != 0 {
		opts = append(opts, llms.WithTemperature(req.Temperature))
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

	choice := resp.Choices[0]
	model := req.Model
	if model == "" {
		model = a.model
	}
	out := &llmgate.Response{
		Content:      choice.Content,
		Model:        model,
		FinishReason: choice.StopReason,
		ToolCalls:    fromLangchainToolCalls(choice),
	}

	// Token usage is reported provider-by-provider under slightly different
	// keys. Try the common ones.
	in := getInt(choice.GenerationInfo, "InputTokens", "PromptTokens", "input_tokens", "prompt_tokens")
	outTok := getInt(choice.GenerationInfo, "OutputTokens", "CompletionTokens", "output_tokens", "completion_tokens")
	out.InputTokens = in
	out.OutputTokens = outTok
	cost, priced := pricing.ComputeWithOK(model, in, outTok)
	out.Usage = llmgate.Usage{
		InputTokens:  in,
		OutputTokens: outTok,
		TotalTokens:  in + outTok,
		CostUSD:      cost,
		Priced:       priced,
	}

	return out, nil
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
// map[string]any, coping with int / int64 / float64 values.
func getInt(m map[string]any, keys ...string) int {
	if m == nil {
		return 0
	}
	for _, k := range keys {
		v, ok := m[k]
		if !ok {
			continue
		}
		switch x := v.(type) {
		case int:
			return x
		case int32:
			return int(x)
		case int64:
			return int(x)
		case uint32:
			return int(x)
		case uint64:
			return int(x)
		case float64:
			return int(x)
		}
	}
	return 0
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
