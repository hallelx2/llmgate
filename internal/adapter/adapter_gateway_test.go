package adapter

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/langchaingo/llms"

	"github.com/hallelx2/llmgate"
)

// TestEmptyResponseIsClassifiedGateway is the HAL-528 regression.
//
// An OpenAI- or Anthropic-compatible gateway that answers HTTP 200 with
// an error envelope arrives here as a successful call with zero choices.
// It used to produce a bare "empty response", classified Unknown, which
// the retry middleware then failed *open* on — burning four paid attempts
// on a permanent configuration fault.
func TestEmptyResponseIsClassifiedGateway(t *testing.T) {
	fm := newFake(&llms.ContentResponse{Choices: nil})
	a := NewAdapter(fm, llmgate.ProviderAnthropic, "glm-4.6", true)

	_, err := a.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("an empty response must be an error")
	}

	if got := llmgate.Classify(err); got != llmgate.ErrClassGateway {
		t.Errorf("Classify = %v, want gateway — Unknown is retried and this fault is permanent", got)
	}

	var llmErr *llmgate.LLMError
	if !errors.As(err, &llmErr) {
		t.Fatalf("error should be a *LLMError, got %T", err)
	}
	if !strings.Contains(err.Error(), "glm-4.6") {
		t.Errorf("error %q should name the model", err)
	}
}

// TestEmptyResponseNamesEndpointAndHint: the z.ai case. A base URL
// missing its version segment is the single most common cause, and the
// engine repo carries a nine-line comment about it — that comment is the
// tell that the error message was not doing its job.
func TestEmptyResponseNamesEndpointAndHint(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		wantHint bool
	}{
		{"missing version segment", "https://api.z.ai/api/anthropic", true},
		{"with version segment", "https://api.z.ai/api/anthropic/v1", false},
		{"trailing slash still versioned", "https://api.z.ai/api/anthropic/v1/", false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fm := newFake(&llms.ContentResponse{Choices: nil})
			a := NewAdapter(fm, llmgate.ProviderAnthropic, "glm-4.6", true)
			a.SetBaseURL(tc.baseURL)

			_, err := a.Complete(context.Background(), llmgate.Request{
				Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "hi"}},
			})
			if err == nil {
				t.Fatal("expected an error")
			}
			if !strings.Contains(err.Error(), tc.baseURL) {
				t.Errorf("error %q should name the endpoint %q", err, tc.baseURL)
			}

			gotHint := strings.Contains(err.Error(), "version segment")
			if gotHint != tc.wantHint {
				t.Errorf("version-segment hint present = %v, want %v (err: %s)", gotHint, tc.wantHint, err)
			}
		})
	}
}

// TestHasVersionSegment pins the detection directly, since the hint is
// only useful if it fires on the right shapes.
func TestHasVersionSegment(t *testing.T) {
	tests := []struct {
		url  string
		want bool
	}{
		{"https://api.anthropic.com/v1", true},
		{"https://api.z.ai/api/anthropic/v1", true},
		{"https://api.z.ai/api/anthropic/v1/", true},
		{"https://api.openai.com/v2/chat", true},
		{"https://api.z.ai/api/anthropic", false},
		{"https://example.com", false},
		{"http://localhost:11434", false},
	}
	for _, tc := range tests {
		if got := hasVersionSegment(tc.url); got != tc.want {
			t.Errorf("hasVersionSegment(%q) = %v, want %v", tc.url, got, tc.want)
		}
	}
}
