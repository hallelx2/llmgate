package llmgate

import (
	"context"
	"fmt"

	lcgoogleai "github.com/tmc/langchaingo/llms/googleai"
)

// GeminiConfig configures the Gemini client.
type GeminiConfig struct {
	APIKey         string
	Model          string
	ReasoningModel string
}

// NewGemini constructs a Client backed by langchaingo's googleai adapter.
//
// Unlike Anthropic and OpenAI, the googleai factory needs a context because
// it authenticates up front. We use a Background context for the client
// construction; per-call contexts still flow into Complete as normal.
func NewGemini(cfg GeminiConfig) (Client, error) {
	opts := []lcgoogleai.Option{}
	if cfg.APIKey != "" {
		opts = append(opts, lcgoogleai.WithAPIKey(cfg.APIKey))
	}
	model := cfg.Model
	if model == "" {
		model = "gemini-2.5-flash"
	}
	opts = append(opts, lcgoogleai.WithDefaultModel(model))

	m, err := lcgoogleai.New(context.Background(), opts...)
	if err != nil {
		return nil, fmt.Errorf("llmgate/gemini: %w", err)
	}
	return &adapter{
		m:        m,
		provider: ProviderGemini,
		model:    model,
		modelSet: cfg.Model != "",
	}, nil
}
