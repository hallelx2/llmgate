package llmgate

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"
)

// AnthropicConfig configures the Anthropic client.
type AnthropicConfig struct {
	APIKey            string
	Model             string
	ReasoningModel    string
	EnablePromptCache bool

	// BaseURL overrides the Anthropic API endpoint. Empty = official.
	BaseURL string
	// HTTPClient is used for requests. Nil => a sensible default.
	HTTPClient *http.Client
	// AnthropicVersion sets the `anthropic-version` header.
	AnthropicVersion string
	// MaxRetries bounds retry attempts for 429 / 5xx (exponential backoff).
	MaxRetries int
}

// Anthropic is a Claude-backed Client backed by the messages API.
type Anthropic struct {
	cfg AnthropicConfig
}

// NewAnthropic constructs a new Anthropic client.
func NewAnthropic(cfg AnthropicConfig) *Anthropic {
	if cfg.Model == "" {
		cfg.Model = "claude-sonnet-4-5"
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com"
	}
	if cfg.AnthropicVersion == "" {
		cfg.AnthropicVersion = "2023-06-01"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 120 * time.Second}
	}
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 3
	}
	return &Anthropic{cfg: cfg}
}

// --- wire types (minimal subset of the messages API) ---

type anthMsgBlock struct {
	Type         string         `json:"type"`
	Text         string         `json:"text,omitempty"`
	CacheControl map[string]any `json:"cache_control,omitempty"`
}

type anthMsg struct {
	Role    string         `json:"role"`
	Content []anthMsgBlock `json:"content"`
}

type anthReq struct {
	Model       string         `json:"model"`
	MaxTokens   int            `json:"max_tokens"`
	Temperature *float64       `json:"temperature,omitempty"`
	System      []anthMsgBlock `json:"system,omitempty"`
	Messages    []anthMsg      `json:"messages"`
}

type anthResp struct {
	ID         string `json:"id"`
	Model      string `json:"model"`
	StopReason string `json:"stop_reason"`
	Content    []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Usage struct {
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	} `json:"usage"`
}

type anthErrEnv struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

type anthCountReq struct {
	Model    string    `json:"model"`
	Messages []anthMsg `json:"messages"`
}

type anthCountResp struct {
	InputTokens int `json:"input_tokens"`
}

// Complete runs a single completion against /v1/messages.
//
// JSONMode is honored via prompt instruction — the Anthropic messages API
// does not yet have a strict JSON-schema mode, so we append a firm
// "reply with only a JSON object matching this schema" nudge and the
// caller validates the result.
func (a *Anthropic) Complete(ctx context.Context, req Request) (*Response, error) {
	if a.cfg.APIKey == "" {
		return nil, fmt.Errorf("anthropic: api key not configured")
	}

	model := req.Model
	if model == "" {
		model = a.cfg.Model
	}
	maxTok := req.MaxTokens
	if maxTok <= 0 {
		maxTok = 4096
	}

	body := anthReq{
		Model:     model,
		MaxTokens: maxTok,
		Messages:  make([]anthMsg, 0, len(req.Messages)),
	}
	if req.Temperature != 0 {
		t := req.Temperature
		body.Temperature = &t
	}

	for _, m := range req.Messages {
		switch m.Role {
		case RoleSystem:
			blk := anthMsgBlock{Type: "text", Text: m.Content}
			if a.cfg.EnablePromptCache {
				blk.CacheControl = map[string]any{"type": "ephemeral"}
			}
			body.System = append(body.System, blk)
		case RoleUser, RoleAssistant:
			body.Messages = append(body.Messages, anthMsg{
				Role:    string(m.Role),
				Content: []anthMsgBlock{{Type: "text", Text: m.Content}},
			})
		}
	}
	if req.JSONMode {
		// Append a JSON nudge on the last user message (or add one).
		nudge := "\n\nRespond with ONLY a single JSON object. No prose, no code fences."
		if len(req.JSONSchema) > 0 {
			nudge += " The object must conform to this JSON schema:\n" + string(req.JSONSchema)
		}
		if n := len(body.Messages); n > 0 && body.Messages[n-1].Role == "user" {
			last := &body.Messages[n-1]
			if len(last.Content) > 0 {
				last.Content[len(last.Content)-1].Text += nudge
			}
		} else {
			body.Messages = append(body.Messages, anthMsg{
				Role:    "user",
				Content: []anthMsgBlock{{Type: "text", Text: nudge}},
			})
		}
	}

	var resp anthResp
	if err := a.doJSON(ctx, "/v1/messages", body, &resp); err != nil {
		return nil, err
	}

	var text bytes.Buffer
	for _, b := range resp.Content {
		if b.Type == "text" {
			text.WriteString(b.Text)
		}
	}

	return &Response{
		Content:      text.String(),
		InputTokens:  resp.Usage.InputTokens + resp.Usage.CacheCreationInputTokens + resp.Usage.CacheReadInputTokens,
		OutputTokens: resp.Usage.OutputTokens,
		Model:        resp.Model,
		FinishReason: resp.StopReason,
	}, nil
}

// CountTokens uses the /v1/messages/count_tokens endpoint for exact counts,
// falling back to a ~4-chars-per-token estimate on error.
func (a *Anthropic) CountTokens(ctx context.Context, text string) (int, error) {
	if a.cfg.APIKey == "" || text == "" {
		return len(text) / 4, nil
	}
	body := anthCountReq{
		Model: a.cfg.Model,
		Messages: []anthMsg{{
			Role:    "user",
			Content: []anthMsgBlock{{Type: "text", Text: text}},
		}},
	}
	var out anthCountResp
	if err := a.doJSON(ctx, "/v1/messages/count_tokens", body, &out); err != nil {
		// Soft-fail: don't break retrieval because the counting endpoint is flaky.
		return len(text) / 4, nil
	}
	return out.InputTokens, nil
}

// doJSON posts body to path, decodes the JSON response, and retries on
// 429 / 5xx with exponential backoff + jitter.
func (a *Anthropic) doJSON(ctx context.Context, path string, body any, out any) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}

	var lastErr error
	for attempt := 0; attempt <= a.cfg.MaxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.cfg.BaseURL+path, bytes.NewReader(raw))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", a.cfg.APIKey)
		req.Header.Set("anthropic-version", a.cfg.AnthropicVersion)

		resp, err := a.cfg.HTTPClient.Do(req)
		if err != nil {
			lastErr = err
			if !sleepBackoff(ctx, attempt) {
				return ctx.Err()
			}
			continue
		}

		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			if out == nil {
				return nil
			}
			return json.Unmarshal(respBody, out)
		}

		var env anthErrEnv
		_ = json.Unmarshal(respBody, &env)
		msg := env.Error.Message
		if msg == "" {
			msg = string(respBody)
		}
		lastErr = fmt.Errorf("anthropic: %s (%d): %s", env.Error.Type, resp.StatusCode, msg)

		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			if !sleepBackoff(ctx, attempt) {
				return ctx.Err()
			}
			continue
		}
		return lastErr
	}
	if lastErr == nil {
		lastErr = errors.New("anthropic: exhausted retries")
	}
	return lastErr
}

// sleepBackoff waits for an exponential+jitter delay or until ctx cancels.
// Returns false if the context cancelled first.
func sleepBackoff(ctx context.Context, attempt int) bool {
	base := time.Duration(1<<attempt) * 500 * time.Millisecond
	if base > 10*time.Second {
		base = 10 * time.Second
	}
	jitter := time.Duration(rand.Int63n(int64(base / 2)))
	select {
	case <-time.After(base + jitter):
		return true
	case <-ctx.Done():
		return false
	}
}
