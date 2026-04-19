// Package anthropic constructs an llmgate.Client backed by langchaingo's
// Anthropic adapter.
package anthropic

import (
	"fmt"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/internal/adapter"
	lcanthropic "github.com/tmc/langchaingo/llms/anthropic"
)

// Config configures the Anthropic client.
//
// ReasoningModel is reserved for a future "deep reason" strategy and isn't
// wired into Complete today — use Request.Model to override per-call.
// EnablePromptCache is preserved on the struct for future work; the current
// langchaingo/llms/anthropic adapter doesn't expose cache_control yet, so
// the flag is a no-op at this layer. Once langchaingo grows the feature
// we'll thread it through.
type Config struct {
	APIKey            string
	Model             string
	ReasoningModel    string
	EnablePromptCache bool

	// BaseURL overrides the Anthropic API endpoint. Empty = official.
	BaseURL string
}

// New constructs an llmgate.Client backed by langchaingo's Anthropic adapter.
func New(cfg Config) (llmgate.Client, error) {
	opts := []lcanthropic.Option{}
	if cfg.APIKey != "" {
		opts = append(opts, lcanthropic.WithToken(cfg.APIKey))
	}
	model := cfg.Model
	if model == "" {
		model = "claude-sonnet-4-5"
	}
	opts = append(opts, lcanthropic.WithModel(model))
	if cfg.BaseURL != "" {
		opts = append(opts, lcanthropic.WithBaseURL(cfg.BaseURL))
	}

	m, err := lcanthropic.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("llmgate/anthropic: %w", err)
	}
	return adapter.NewAdapter(m, llmgate.ProviderAnthropic, model, cfg.Model != ""), nil
}
