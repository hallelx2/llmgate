// Package openai constructs an llmgate.Client backed by langchaingo's
// OpenAI adapter.
package openai

import (
	"fmt"

	lcopenai "github.com/tmc/langchaingo/llms/openai"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/internal/adapter"
)

// Config configures the OpenAI client.
type Config struct {
	APIKey         string
	Model          string
	ReasoningModel string

	// BaseURL overrides the OpenAI API endpoint. Handy for Azure OpenAI,
	// LM Studio, Ollama-openai-compat, etc.
	BaseURL string
}

// New constructs an llmgate.Client backed by langchaingo's OpenAI adapter.
func New(cfg Config) (llmgate.Client, error) {
	opts := []lcopenai.Option{}
	if cfg.APIKey != "" {
		opts = append(opts, lcopenai.WithToken(cfg.APIKey))
	}
	model := cfg.Model
	if model == "" {
		model = "gpt-4o-mini"
	}
	opts = append(opts, lcopenai.WithModel(model))
	if cfg.BaseURL != "" {
		opts = append(opts, lcopenai.WithBaseURL(cfg.BaseURL))
	}

	m, err := lcopenai.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("llmgate/openai: %w", err)
	}
	return adapter.NewAdapter(m, llmgate.ProviderOpenAI, model, cfg.Model != ""), nil
}
