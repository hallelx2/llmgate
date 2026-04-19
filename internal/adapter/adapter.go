// Package adapter is the internal seam where llmgate's Client interface
// meets langchaingo's provider implementations. Providers construct an
// Adapter via NewAdapter; callers never see this package directly.
package adapter

import (
	"context"
	"fmt"

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

	resp, err := a.m.GenerateContent(ctx, msgs, opts...)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", a.provider, err)
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("%s: empty response", a.provider)
	}

	choice := resp.Choices[0]
	out := &llmgate.Response{
		Content:      choice.Content,
		Model:        req.Model,
		FinishReason: choice.StopReason,
	}
	if out.Model == "" {
		out.Model = a.model
	}

	// Token usage is reported provider-by-provider under slightly different
	// keys. Try the common ones.
	in := getInt(choice.GenerationInfo, "InputTokens", "PromptTokens", "input_tokens", "prompt_tokens")
	outTok := getInt(choice.GenerationInfo, "OutputTokens", "CompletionTokens", "output_tokens", "completion_tokens")
	out.InputTokens = in
	out.OutputTokens = outTok
	out.Usage = llmgate.Usage{
		InputTokens:  in,
		OutputTokens: outTok,
		TotalTokens:  in + outTok,
		CostUSD:      pricing.Compute(out.Model, in, outTok),
	}

	return out, nil
}

// Capabilities reports known capabilities for the adapter's configured
// model by looking them up in the registry. Satisfies capabilities.Capable.
func (a *Adapter) Capabilities() capabilities.Capabilities {
	return capabilities.Lookup(a.model)
}

// CountTokens falls back to the estimator installed by the provider factory,
// or a ~4-chars-per-token guess.
func (a *Adapter) CountTokens(ctx context.Context, text string) (int, error) {
	if a.countTok != nil {
		return a.countTok(ctx, text)
	}
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
		out = append(out, llms.MessageContent{
			Role:  role,
			Parts: []llms.ContentPart{llms.TextContent{Text: m.Content}},
		})
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
