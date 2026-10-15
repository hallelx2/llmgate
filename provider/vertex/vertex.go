// Package vertex constructs an llmgate.Client that calls Gemini through
// Google Cloud Vertex AI.
//
// It talks to the Vertex REST API directly rather than through the
// langchaingo googleai adapter the gemini package uses, for three reasons
// the adapter cannot meet:
//
//   - Authentication is Application Default Credentials (a VM's service
//     account, or `gcloud auth application-default login` on a laptop), so
//     usage bills to a Google Cloud project — and to that project's credits
//     — instead of an AI Studio key.
//   - Thinking is controllable per call (thinkingBudget or thinkingLevel),
//     and the thinking tokens the model spends are reported. Gemini 3 bills
//     thinking as output, so an uncapped call costs more and takes longer
//     than the answer alone.
//   - Usage is read from usageMetadata in full: cached prompt tokens,
//     candidate tokens and thought tokens, normalised to llmgate's disjoint
//     Usage form and priced from the pricing book.
//
// Gemini 3 attaches a thoughtSignature to each function call and requires
// it back on the next turn. llmgate's ToolCall carries no signature field,
// so the signature travels inside ToolCall.ID (see encodeCallID). Callers
// already echo IDs back verbatim, which is all the round trip needs.
package vertex

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/pricing"
)

// DefaultModel is used when neither Config.Model nor Request.Model names one.
const DefaultModel = "gemini-2.5-flash-lite"

// DefaultLocation is the Vertex location used when none is configured.
// "global" routes to any region with capacity and is the cheaper price
// tier for Gemini 3 (regional endpoints list at about 10% more).
const DefaultLocation = "global"

const cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

// Thinking controls how much the model reasons before it answers.
//
// Gemini 2.5 takes a token budget; Gemini 3 takes a level and, on some
// models, also honours a budget. Set at most one of Budget and Level. A
// zero Thinking leaves the model's own default in place, which on most
// Gemini 3 models is to think.
type Thinking struct {
	// Budget is the maximum number of thinking tokens. A pointer so that
	// 0 ("do not think") is distinct from unset. Not every model accepts
	// 0: gemini-3.8-flash keeps thinking regardless (measured 2026-10-16).
	Budget *int

	// Level is a Gemini 3 thinking level: "minimal", "low", "medium" or
	// "high". Not every model accepts every level.
	Level string

	// IncludeThoughts asks for thought summaries, returned in
	// Response.ReasoningContent.
	IncludeThoughts bool
}

func (t Thinking) isZero() bool {
	return t.Budget == nil && t.Level == "" && !t.IncludeThoughts
}

// Config configures the Vertex client.
type Config struct {
	// Project is the Google Cloud project billed for the calls. Empty
	// falls back to GOOGLE_CLOUD_PROJECT, then to the project the
	// credentials name.
	Project string

	// Location is the Vertex location: "global" (default) or a region
	// such as "us-central1". Empty falls back to GOOGLE_CLOUD_LOCATION.
	Location string

	// Model is the default model ID, e.g. "gemini-2.5-flash-lite".
	// Request.Model overrides it per call.
	Model string

	// Thinking is applied to every call. Hold it fixed across the systems
	// in a comparison so none of them gets more deliberation than another.
	Thinking Thinking

	// TokenSource supplies OAuth2 access tokens. Nil uses Application
	// Default Credentials with the cloud-platform scope.
	TokenSource oauth2.TokenSource

	// HTTPClient sends the requests. Nil uses a client with Timeout.
	HTTPClient *http.Client

	// Timeout bounds each request when HTTPClient is nil. Default 120s.
	Timeout time.Duration

	// BaseURL overrides the endpoint root, for tests. Empty derives it
	// from Location.
	BaseURL string
}

// Client is a Vertex-backed llmgate.Client.
type Client struct {
	project  string
	location string
	model    string
	thinking Thinking
	ts       oauth2.TokenSource
	http     *http.Client
	baseURL  string
}

var _ llmgate.Client = (*Client)(nil)

// New constructs a Vertex client. It resolves credentials up front, so a
// machine with no Application Default Credentials fails here rather than
// on the first call.
func New(ctx context.Context, cfg Config) (*Client, error) {
	location := firstNonEmpty(cfg.Location, os.Getenv("GOOGLE_CLOUD_LOCATION"), DefaultLocation)
	project := firstNonEmpty(cfg.Project, os.Getenv("GOOGLE_CLOUD_PROJECT"))

	ts := cfg.TokenSource
	if ts == nil {
		// A pre-minted token, as Terraform's Google provider accepts it:
		// `export GOOGLE_OAUTH_ACCESS_TOKEN=$(gcloud auth print-access-token)`
		// works on a machine with a gcloud login but no ADC file. Tokens
		// last about an hour; long runs should use ADC instead.
		if tok := os.Getenv("GOOGLE_OAUTH_ACCESS_TOKEN"); tok != "" {
			ts = oauth2.StaticTokenSource(&oauth2.Token{AccessToken: tok})
		}
	}
	if ts == nil {
		creds, err := google.FindDefaultCredentials(ctx, cloudPlatformScope)
		if err != nil {
			return nil, fmt.Errorf("llmgate/vertex: no Application Default Credentials (run `gcloud auth application-default login`, or attach a service account): %w", err)
		}
		ts = creds.TokenSource
		if project == "" {
			project = creds.ProjectID
		}
	}
	if project == "" {
		return nil, errors.New("llmgate/vertex: no project: set Config.Project or GOOGLE_CLOUD_PROJECT")
	}
	if cfg.Thinking.Budget != nil && cfg.Thinking.Level != "" {
		return nil, errors.New("llmgate/vertex: set Thinking.Budget or Thinking.Level, not both")
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		timeout := cfg.Timeout
		if timeout <= 0 {
			timeout = 120 * time.Second
		}
		httpClient = &http.Client{Timeout: timeout}
	}

	base := strings.TrimRight(cfg.BaseURL, "/")
	if base == "" {
		base = endpointRoot(location)
	}

	return &Client{
		project:  project,
		location: location,
		model:    firstNonEmpty(cfg.Model, DefaultModel),
		thinking: cfg.Thinking,
		ts:       oauth2.ReuseTokenSource(nil, ts),
		http:     httpClient,
		baseURL:  base,
	}, nil
}

// endpointRoot is the API host for a location: the global endpoint has no
// region prefix, every other location does.
func endpointRoot(location string) string {
	if location == "global" {
		return "https://aiplatform.googleapis.com"
	}
	return "https://" + location + "-aiplatform.googleapis.com"
}

// Project reports the billed project.
func (c *Client) Project() string { return c.project }

// Location reports the Vertex location.
func (c *Client) Location() string { return c.location }

// Model reports the default model.
func (c *Client) Model() string { return c.model }

func (c *Client) modelURL(model, method string) string {
	return fmt.Sprintf("%s/v1/projects/%s/locations/%s/publishers/google/models/%s:%s",
		c.baseURL, c.project, c.location, model, method)
}

// Complete runs one generateContent call.
func (c *Client) Complete(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
	model := firstNonEmpty(req.Model, c.model)
	body, err := buildRequest(req, c.thinking)
	if err != nil {
		return nil, err
	}

	var wire generateResponse
	if err := c.post(ctx, c.modelURL(model, "generateContent"), body, &wire); err != nil {
		return nil, err
	}
	return c.toResponse(model, wire)
}

// CountTokens asks Vertex's countTokens endpoint for the model's own count.
func (c *Client) CountTokens(ctx context.Context, text string) (int, error) {
	body := map[string]any{
		"contents": []content{{Role: "user", Parts: []part{{Text: text}}}},
	}
	var wire struct {
		TotalTokens int `json:"totalTokens"`
	}
	if err := c.post(ctx, c.modelURL(c.model, "countTokens"), body, &wire); err != nil {
		return 0, err
	}
	return wire.TotalTokens, nil
}

func (c *Client) post(ctx context.Context, url string, payload any, out any) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("llmgate/vertex: encode request: %w", err)
	}
	tok, err := c.ts.Token()
	if err != nil {
		return &llmgate.LLMError{Class: llmgate.ErrClassAuth, Provider: llmgate.ProviderGemini,
			Message: "vertex: fetch access token: " + err.Error(), Cause: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return fmt.Errorf("llmgate/vertex: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+tok.AccessToken)
	// Bill quota to the configured project even when the credentials
	// belong to a different one (a user account, a shared SA).
	httpReq.Header.Set("x-goog-user-project", c.project)

	resp, err := c.http.Do(httpReq)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		return &llmgate.LLMError{Class: llmgate.ErrClassTransient, Provider: llmgate.ProviderGemini,
			Message: "vertex: " + err.Error(), Cause: err}
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return &llmgate.LLMError{Class: llmgate.ErrClassTransient, Provider: llmgate.ProviderGemini,
			Message: "vertex: read response: " + err.Error(), Cause: err}
	}
	if resp.StatusCode != http.StatusOK {
		return httpError(resp, respBody)
	}
	if err := json.Unmarshal(respBody, out); err != nil {
		return &llmgate.LLMError{Class: llmgate.ErrClassGateway, Provider: llmgate.ProviderGemini,
			StatusCode: resp.StatusCode, Message: "vertex: decode response: " + err.Error(), Cause: err}
	}
	return nil
}

func (c *Client) toResponse(model string, wire generateResponse) (*llmgate.Response, error) {
	if wire.PromptFeedback != nil && wire.PromptFeedback.BlockReason != "" {
		return nil, &llmgate.LLMError{Class: llmgate.ErrClassContent, Provider: llmgate.ProviderGemini,
			Message: "vertex: prompt blocked: " + wire.PromptFeedback.BlockReason}
	}
	if len(wire.Candidates) == 0 {
		return nil, &llmgate.LLMError{Class: llmgate.ErrClassGateway, Provider: llmgate.ProviderGemini,
			Message: "vertex: response has no candidates"}
	}
	cand := wire.Candidates[0]

	out := &llmgate.Response{Model: firstNonEmpty(wire.ModelVersion, model), FinishReason: cand.FinishReason}
	var text, thoughts strings.Builder
	for i, p := range cand.Content.Parts {
		switch {
		case p.FunctionCall != nil:
			args := p.FunctionCall.Args
			if len(args) == 0 {
				args = json.RawMessage("{}")
			}
			out.ToolCalls = append(out.ToolCalls, llmgate.ToolCall{
				ID:    encodeCallID(p.FunctionCall.ID, i, p.ThoughtSignature),
				Name:  p.FunctionCall.Name,
				Input: []byte(args),
			})
		case p.Thought:
			thoughts.WriteString(p.Text)
		default:
			text.WriteString(p.Text)
		}
	}
	out.Content = text.String()
	out.ReasoningContent = thoughts.String()

	switch cand.FinishReason {
	case "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION":
		if out.Content == "" && len(out.ToolCalls) == 0 {
			return nil, &llmgate.LLMError{Class: llmgate.ErrClassContent, Provider: llmgate.ProviderGemini,
				Message: "vertex: response blocked: " + cand.FinishReason}
		}
	}

	out.Usage = usageFor(out.Model, model, c.location, wire.UsageMetadata)
	out.InputTokens = out.Usage.InputTokens
	out.OutputTokens = out.Usage.OutputTokens
	return out, nil
}

// usageFor normalises usageMetadata to llmgate's disjoint Usage.
//
// Vertex reports cached tokens inside promptTokenCount and thought tokens
// beside candidatesTokenCount; llmgate wants Input to exclude the cache
// and Output to include reasoning.
func usageFor(servedModel, requestedModel, location string, um *usageMetadata) llmgate.Usage {
	if um == nil {
		return llmgate.Usage{}
	}
	cached := min(um.CachedContentTokenCount, um.PromptTokenCount)
	u := llmgate.Usage{
		InputTokens:     um.PromptTokenCount - cached,
		CacheReadTokens: cached,
		OutputTokens:    um.CandidatesTokenCount + um.ThoughtsTokenCount,
		ReasoningTokens: um.ThoughtsTokenCount,
		TotalTokens:     um.TotalTokenCount,
		TokensReported:  true,
	}
	if u.TotalTokens == 0 {
		u.TotalTokens = um.PromptTokenCount + u.OutputTokens + um.ToolUsePromptTokenCount
	}
	// Tool-use prompt tokens are billed as input.
	u.InputTokens += um.ToolUsePromptTokenCount

	tk := pricing.Tokens{Input: u.InputTokens, Output: u.OutputTokens, CacheRead: u.CacheReadTokens, Reasoning: u.ReasoningTokens}
	// The served model version ("gemini-2.5-flash-lite-001") may carry a
	// suffix the price book resolves by prefix; try it first, then the ID
	// the caller asked for. Lookup before Compute so a miss on the first
	// does not fire the unpriced-model warning for a model that is priced.
	priceAs := requestedModel
	if _, ok := pricing.Lookup(servedModel); ok {
		priceAs = servedModel
	}
	if cost, ok := pricing.ComputeTokens(priceAs, tk); ok {
		u.CostUSD, u.Priced = cost, true
		u.CostUSD *= regionalUplift(priceAs, location)
	}
	return u
}

// regionalUplift is the multiplier Vertex applies to a model's price off
// the global endpoint. Gemini 3 models list regional ("non-global")
// prices 10% above global; Gemini 2.x has one price everywhere. Measured
// on cloud.google.com/vertex-ai/generative-ai/pricing, 2026-10-16.
func regionalUplift(model, location string) float64 {
	if location == "global" || location == "" {
		return 1
	}
	if strings.HasPrefix(pricing.Canonical(model), "gemini-3") {
		return 1.1
	}
	return 1
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
