// Package gemini constructs an llmgate.Client backed by langchaingo's
// googleai adapter.
package gemini

import (
	"context"
	"fmt"

	lcgoogleai "github.com/tmc/langchaingo/llms/googleai"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/internal/adapter"
)

// Config configures the Gemini client.
type Config struct {
	APIKey         string
	Model          string
	ReasoningModel string
}

// New constructs an llmgate.Client backed by langchaingo's googleai adapter.
//
// Unlike Anthropic and OpenAI, the googleai factory needs a context because
// it authenticates up front. We use a Background context for the client
// construction; per-call contexts still flow into Complete as normal.
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

	m, err := lcgoogleai.New(context.Background(), opts...)
	if err != nil {
		return nil, fmt.Errorf("llmgate/gemini: %w", err)
	}
	return adapter.NewAdapter(m, llmgate.ProviderGemini, model, cfg.Model != ""), nil
}
