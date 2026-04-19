package budget_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/middleware/budget"
)

func TestEnforcesTotalCap(t *testing.T) {
	inner := &llmgate.Mock{
		Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
			return &llmgate.Response{Usage: llmgate.Usage{CostUSD: 0.5}}, nil
		},
	}
	client := budget.New(budget.Config{TotalUSD: 1.0})(inner)

	if _, err := client.Complete(context.Background(), llmgate.Request{}); err != nil {
		t.Fatalf("first: %v", err)
	}
	if _, err := client.Complete(context.Background(), llmgate.Request{}); err != nil {
		t.Fatalf("second: %v", err)
	}
	_, err := client.Complete(context.Background(), llmgate.Request{})
	if !errors.Is(err, budget.ErrExceeded) {
		t.Fatalf("third call err = %v, want ErrExceeded", err)
	}
}

func TestDailyRollover(t *testing.T) {
	now := time.Date(2026, 4, 19, 23, 30, 0, 0, time.UTC)
	clock := &now
	inner := &llmgate.Mock{
		Respond: func(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
			return &llmgate.Response{Usage: llmgate.Usage{CostUSD: 0.9}}, nil
		},
	}
	client := budget.New(budget.Config{
		DailyUSD: 1.0,
		Now:      func() time.Time { return *clock },
	})(inner)

	// Spend 0.9 on day 1.
	if _, err := client.Complete(context.Background(), llmgate.Request{}); err != nil {
		t.Fatalf("day1 first: %v", err)
	}
	// Second call still allowed pre-check (0.9 < 1.0); it pushes us to 1.8.
	if _, err := client.Complete(context.Background(), llmgate.Request{}); err != nil {
		t.Fatalf("day1 second: %v", err)
	}
	// Third call refused (1.8 >= 1.0).
	if _, err := client.Complete(context.Background(), llmgate.Request{}); !errors.Is(err, budget.ErrExceeded) {
		t.Fatalf("day1 third err = %v, want ErrExceeded", err)
	}

	// Roll the clock to next UTC day.
	*clock = now.Add(24 * time.Hour)

	if _, err := client.Complete(context.Background(), llmgate.Request{}); err != nil {
		t.Fatalf("day2 first: %v", err)
	}
}

func TestUnlimited(t *testing.T) {
	inner := &llmgate.Mock{Reply: "ok"}
	client := budget.New(budget.Config{})(inner)
	for i := 0; i < 5; i++ {
		if _, err := client.Complete(context.Background(), llmgate.Request{}); err != nil {
			t.Fatalf("unlimited %d: %v", i, err)
		}
	}
}
