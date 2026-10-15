package vertex

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
)

// TestLive calls the real Vertex API. It runs only with VERTEX_LIVE=1 and
// Application Default Credentials for a project with aiplatform enabled:
//
//	VERTEX_LIVE=1 GOOGLE_CLOUD_PROJECT=vectorless-prod go test ./provider/vertex -run Live -v
func TestLive(t *testing.T) {
	if os.Getenv("VERTEX_LIVE") != "1" {
		t.Skip("set VERTEX_LIVE=1 to call Vertex")
	}
	models := []string{"gemini-2.5-flash-lite", "gemini-3.5-flash-lite"}
	if m := os.Getenv("VERTEX_LIVE_MODEL"); m != "" {
		models = []string{m}
	}
	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			c, err := New(ctx, Config{Model: model, Thinking: Thinking{Budget: llmgate.Int(0)}})
			if err != nil {
				t.Fatal(err)
			}
			start := time.Now()
			resp, err := c.Complete(ctx, llmgate.Request{
				Messages:    []llmgate.Message{{Role: llmgate.RoleUser, Content: "What is 17*23? Reply with the number only."}},
				MaxTokens:   32,
				Temperature: llmgate.Float64(0),
			})
			if err != nil {
				t.Fatal(err)
			}
			u := resp.Usage
			t.Logf("%s: %q in %v; input=%d output=%d reasoning=%d cost=$%.8f priced=%v",
				resp.Model, resp.Content, time.Since(start).Round(time.Millisecond),
				u.InputTokens, u.OutputTokens, u.ReasoningTokens, u.CostUSD, u.Priced)
			if resp.Content == "" || !u.TokensReported || !u.Priced {
				t.Errorf("want content, reported tokens and a price; got %+v", resp)
			}
			if u.ReasoningTokens != 0 {
				t.Errorf("thinking budget 0 still spent %d thinking tokens", u.ReasoningTokens)
			}

			n, err := c.CountTokens(ctx, "hello world, this is a token count")
			if err != nil || n == 0 {
				t.Errorf("CountTokens = %d, %v", n, err)
			}

			// A tool round trip, which on Gemini 3 needs the thought
			// signature echoed back.
			tools := []llmgate.ToolDef{{Name: "weather", Description: "Get the weather for a city",
				InputSchema: []byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)}}
			first, err := c.Complete(ctx, llmgate.Request{
				Messages:   []llmgate.Message{{Role: llmgate.RoleUser, Content: "What's the weather in Lagos?"}},
				Tools:      tools,
				ToolChoice: "required",
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(first.ToolCalls) == 0 {
				t.Fatalf("no tool call: %+v", first)
			}
			call := first.ToolCalls[0]
			second, err := c.Complete(ctx, llmgate.Request{
				Messages: []llmgate.Message{
					{Role: llmgate.RoleUser, Content: "What's the weather in Lagos?"},
					{Role: llmgate.RoleAssistant, ToolCalls: first.ToolCalls},
					{Role: llmgate.RoleTool, ToolCallID: call.ID, Content: `{"temp_c":31,"sky":"clear"}`},
				},
				Tools: tools,
			})
			if err != nil {
				t.Fatalf("second turn: %v", err)
			}
			t.Logf("tool round trip answer: %q", second.Content)
			if second.Content == "" {
				t.Error("empty answer after the tool result")
			}
		})
	}
}
