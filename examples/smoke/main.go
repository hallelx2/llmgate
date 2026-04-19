// Command smoke is a one-shot live check against a real provider.
// Run with: GEMINI_API_KEY=... go run ./examples/smoke
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/capabilities"
	"github.com/hallelx2/llmgate/middleware/retry"
	"github.com/hallelx2/llmgate/provider/gemini"
)

func main() {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		fmt.Fprintln(os.Stderr, "GEMINI_API_KEY not set")
		os.Exit(1)
	}

	client, err := gemini.New(gemini.Config{APIKey: key})
	if err != nil {
		fmt.Fprintln(os.Stderr, "construct:", err)
		os.Exit(1)
	}

	// Wrap in retry middleware to exercise it against the real network.
	client = retry.New(retry.Config{MaxRetries: 2})(client)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := client.Complete(ctx, llmgate.Request{
		Messages: []llmgate.Message{
			{Role: llmgate.RoleUser, Content: "In one sentence: what is vectorless retrieval?"},
		},
		MaxTokens:   1024,
		Temperature: 0.2,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "complete:", err)
		os.Exit(1)
	}

	fmt.Println("--- reply ---")
	fmt.Println(resp.Content)
	fmt.Println("--- meta ---")
	fmt.Printf("model=%q finish=%q\n", resp.Model, resp.FinishReason)
	fmt.Printf("usage: in=%d out=%d total=%d cost=$%.6f\n",
		resp.Usage.InputTokens, resp.Usage.OutputTokens,
		resp.Usage.TotalTokens, resp.Usage.CostUSD)

	if c, ok := client.(capabilities.Capable); ok {
		caps := c.Capabilities()
		fmt.Printf("caps: ctx=%d json=%v stream=%v tools=%v vision=%v\n",
			caps.MaxContext, caps.SupportsJSONMode,
			caps.SupportsStreaming, caps.SupportsTools, caps.SupportsVision)
	}
}
