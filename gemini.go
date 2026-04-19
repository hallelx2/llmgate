package llmgate

import "context"

// GeminiConfig configures the Gemini client.
type GeminiConfig struct {
	APIKey         string
	Model          string
	ReasoningModel string
}

// Gemini is a Google-backed Client.
type Gemini struct{ cfg GeminiConfig }

// NewGemini constructs a new Gemini client.
func NewGemini(cfg GeminiConfig) *Gemini {
	if cfg.Model == "" {
		cfg.Model = "gemini-2.0-flash"
	}
	return &Gemini{cfg: cfg}
}

func (g *Gemini) Complete(ctx context.Context, req Request) (*Response, error) {
	// TODO(phase-1): swap to a langchaingo/llms/googleai adapter.
	return nil, ErrNotImplemented
}

func (g *Gemini) CountTokens(ctx context.Context, text string) (int, error) {
	return len(text) / 4, nil
}
