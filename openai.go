package llmgate

import (
	"fmt"

	lcopenai "github.com/tmc/langchaingo/llms/openai"
)

// OpenAIConfig configures the OpenAI client.
type OpenAIConfig struct {
	APIKey         string
	Model          string
	ReasoningModel string

	// BaseURL overrides the OpenAI API endpoint. Handy for Azure OpenAI,
	// LM Studio, Ollama-openai-compat, etc.
	BaseURL string
}

// NewOpenAI constructs a Client backed by langchaingo's OpenAI adapter.
func NewOpenAI(cfg OpenAIConfig) (Client, error) {
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
	return &adapter{
		m:        m,
		provider: ProviderOpenAI,
		model:    model,
		modelSet: cfg.Model != "",
	}, nil
}
