// Package anthropic constructs an llmgate.Client backed by langchaingo's
// Anthropic adapter.
package anthropic

import (
	"fmt"

	lcanthropic "github.com/tmc/langchaingo/llms/anthropic"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/internal/adapter"
)

// Config configures the Anthropic client.
//
// ReasoningModel is reserved for a future "deep reason" strategy and isn't
// wired into Complete today — use Request.Model to override per-call.
type Config struct {
	APIKey         string
	Model          string
	ReasoningModel string

	// EnablePromptCache caches the first user message of every request.
	//
	// It is a shorthand for the common shape, where a call opens with a
	// large stable block — a document, a schema, a long brief — followed by
	// the part that varies. Anthropic bills a cache read at a tenth of the
	// input rate, so a 120k-token prompt whose first 100k are cached costs
	// roughly a third of the uncached call.
	//
	// Set Message.CacheBreakpoint yourself when the prefix does not end at
	// the first message; an explicit breakpoint anywhere in the request
	// turns this shorthand off, so the two never fight. Caching below the
	// provider's minimum cacheable length (1024 tokens on most models)
	// silently does nothing.
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
	a := adapter.NewAdapter(m, llmgate.ProviderAnthropic, model, cfg.Model != "")
	a.SetPromptCache(cfg.EnablePromptCache)
	a.SetBaseURL(cfg.BaseURL)
	return a, nil
}
