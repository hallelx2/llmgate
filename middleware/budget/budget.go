// Package budget provides a USD-cost budget middleware for llmgate.Client
// that refuses calls once a daily or all-time cap is hit.
package budget

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/capabilities"
)

// Config configures New.
type Config struct {
	// DailyUSD is the per-UTC-day cap. <=0 means unlimited.
	DailyUSD float64

	// TotalUSD is the all-time cap. <=0 means unlimited.
	TotalUSD float64

	// Now overrides time.Now for tests; nil uses time.Now.
	Now func() time.Time
}

// ErrExceeded is returned by Complete when a cap is hit.
var ErrExceeded = errors.New("llmgate/budget: budget exceeded")

// New returns a Middleware that refuses calls once a cap is hit.
// Costs are counted from Response.Usage.CostUSD after each successful call.
// Calls with zero CostUSD (unpriced models) still consume but cost 0.
//
// The cap is enforced as a post-hoc counter: a single call may push the
// counters past the cap, but the next call is refused with ErrExceeded.
// The daily counter resets at UTC midnight.
func New(cfg Config) llmgate.Middleware {
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	shared := &budgetClient{cfg: cfg, now: now, dayStart: utcDate(now())}
	return func(inner llmgate.Client) llmgate.Client {
		shared.inner = inner
		return shared
	}
}

type budgetClient struct {
	cfg   Config
	now   func() time.Time
	inner llmgate.Client

	mu       sync.Mutex
	daily    float64
	total    float64
	dayStart time.Time
}

func (b *budgetClient) Complete(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
	if err := b.check(); err != nil {
		return nil, err
	}
	resp, err := b.inner.Complete(ctx, req)
	if err != nil {
		return resp, err
	}
	b.add(resp.Usage.CostUSD)
	return resp, nil
}

func (b *budgetClient) CountTokens(ctx context.Context, text string) (int, error) {
	return b.inner.CountTokens(ctx, text)
}

// Capabilities delegates to the inner client.
func (b *budgetClient) Capabilities() capabilities.Capabilities { return capabilities.Of(b.inner) }

func (b *budgetClient) check() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.rolloverLocked()
	if b.cfg.TotalUSD > 0 && b.total >= b.cfg.TotalUSD {
		return ErrExceeded
	}
	if b.cfg.DailyUSD > 0 && b.daily >= b.cfg.DailyUSD {
		return ErrExceeded
	}
	return nil
}

func (b *budgetClient) add(cost float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.rolloverLocked()
	b.daily += cost
	b.total += cost
}

func (b *budgetClient) rolloverLocked() {
	today := utcDate(b.now())
	if !today.Equal(b.dayStart) {
		b.dayStart = today
		b.daily = 0
	}
}

func utcDate(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}
