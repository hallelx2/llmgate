// Package gemini constructs an llmgate.Client backed by langchaingo's
// googleai adapter.
package gemini

import (
	"context"
	"fmt"
	"time"

	lcgoogleai "github.com/tmc/langchaingo/llms/googleai"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/internal/adapter"
)

// Config configures the Gemini client.
type Config struct {
	APIKey string
	Model  string

	// ReasoningModel is reserved for a future "deep reasoning" strategy.
	// Not wired into Complete today — use Request.Model to override
	// per-call. See Anthropic/OpenAI Config for the same pattern.
	ReasoningModel string

	// AuthTimeout is the maximum time to wait for Gemini authentication
	// during client construction. Default 30s. Prevents indefinite hangs
	// when Google's endpoint is unreachable.
	AuthTimeout time.Duration
}

// New constructs an llmgate.Client backed by langchaingo's googleai adapter.
//
// Unlike Anthropic and OpenAI, the googleai factory needs a context because
// it authenticates up front. We use a timeout-bounded context so construction
// can't hang indefinitely if Google's endpoint is down. Per-call contexts
// still flow into Complete as normal.
func New(cfg Config) (llmgate.Client, error) {
	opts := []lcgoogleai.Option{}
	if cfg.APIKey != "" {
		opts = append(opts, lcgoogleai.WithAPIKey(cfg.APIKey))
	}
	model := cfg.Model
	if model == "" {
		model = "gemini-2.5-flash"
	}
	opts = append(opts, lcgoogleai.WithDefaultModel(model))

	authTimeout := cfg.AuthTimeout
	if authTimeout <= 0 {
		authTimeout = 30 * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), authTimeout)
	defer cancel()

	m, err := lcgoogleai.New(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("llmgate/gemini: %w", err)
	}
	return adapter.NewAdapter(m, llmgate.ProviderGemini, model, cfg.Model != ""), nil
}
