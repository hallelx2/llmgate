// Package budget provides a USD-cost budget middleware for llmgate.Client
// that refuses calls once a daily or all-time cap is hit.
package budget

import (
	"context"
	"errors"
	"math"
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

	// RefuseUnpriced rejects calls whose model has no price-book entry.
	//
	// An unpriced call debits nothing, so with this false a budget over
	// unpriced models is silently unlimited — the cap never trips no
	// matter how much is actually spent. Set it true when the cap is a
	// real spending limit rather than a rough guard.
	RefuseUnpriced bool

	// Now overrides time.Now for tests; nil uses time.Now.
	Now func() time.Time
}

// ErrExceeded is returned by Complete when a cap is hit.
var ErrExceeded = errors.New("llmgate/budget: budget exceeded")

// ErrUnpriced is returned when RefuseUnpriced is set and a call's model
// has no price-book entry, so its spend could not be counted.
var ErrUnpriced = errors.New("llmgate/budget: model is unpriced, cost cannot be enforced")

// New returns a Middleware that refuses calls once a cap is hit.
//
// Costs are counted from Response.Usage.CostUSD after each successful
// call. Estimated costs count too — an approximation tracks a budget far
// better than a zero does.
//
// Applying the returned Middleware to several clients gives each its own
// wrapper over one shared Ledger, so a router's providers draw down a
// single budget without aliasing each other.
//
// The cap is enforced as a post-hoc counter: a single call may push the
// counters past the cap, but the next call is refused with ErrExceeded.
// The daily counter resets at UTC midnight.
func New(cfg Config) llmgate.Middleware {
	l := NewLedger(cfg)
	return func(inner llmgate.Client) llmgate.Client {
		return &budgetClient{inner: inner, ledger: l}
	}
}

// NewWithLedger is New against a Ledger the caller already holds, so
// spend can be inspected without threading the middleware around.
func NewWithLedger(l *Ledger) llmgate.Middleware {
	return func(inner llmgate.Client) llmgate.Client {
		return &budgetClient{inner: inner, ledger: l}
	}
}

// Ledger holds the shared spend counters behind one or more budget
// clients. Safe for concurrent use.
type Ledger struct {
	cfg Config
	now func() time.Time

	mu       sync.Mutex
	daily    float64
	total    float64
	dayStart time.Time
}

// NewLedger creates a spend ledger. Share one across several wrapped
// clients to enforce a single budget over all of them.
func NewLedger(cfg Config) *Ledger {
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	return &Ledger{cfg: cfg, now: now, dayStart: utcDate(now())}
}

// Spent reports spend so far against the daily and all-time counters.
func (l *Ledger) Spent() (daily, total float64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.rolloverLocked()
	return l.daily, l.total
}

// Remaining reports headroom under each cap. An unset cap reports
// math.Inf(1) — there is no limit to run out of.
func (l *Ledger) Remaining() (daily, total float64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.rolloverLocked()
	return remaining(l.cfg.DailyUSD, l.daily), remaining(l.cfg.TotalUSD, l.total)
}

func remaining(limit, spent float64) float64 {
	if limit <= 0 {
		return math.Inf(1)
	}
	if r := limit - spent; r > 0 {
		return r
	}
	return 0
}

type budgetClient struct {
	inner  llmgate.Client
	ledger *Ledger
}

// Complete rejects the call with ErrExceeded when the configured cap has been
// reached; otherwise it forwards to the inner client and debits the cost.
func (b *budgetClient) Complete(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
	if err := b.ledger.check(); err != nil {
		return nil, err
	}
	resp, err := b.inner.Complete(ctx, req)
	if err != nil {
		return resp, err
	}
	if b.ledger.cfg.RefuseUnpriced && !resp.Usage.Priced {
		// The call already happened and cannot be un-spent, but refusing
		// here stops an unbounded run of uncountable calls.
		return resp, ErrUnpriced
	}
	b.ledger.add(resp.Usage.CostUSD)
	return resp, nil
}

// CountTokens passes through to the inner client without touching the budget.
func (b *budgetClient) CountTokens(ctx context.Context, text string) (int, error) {
	return b.inner.CountTokens(ctx, text)
}

// Capabilities delegates to the inner client.
func (b *budgetClient) Capabilities() capabilities.Capabilities {
	return capabilities.Of(b.inner)
}

// Ledger exposes the shared counters this client debits.
func (b *budgetClient) Ledger() *Ledger { return b.ledger }

func (l *Ledger) check() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.rolloverLocked()
	if l.cfg.TotalUSD > 0 && l.total >= l.cfg.TotalUSD {
		return ErrExceeded
	}
	if l.cfg.DailyUSD > 0 && l.daily >= l.cfg.DailyUSD {
		return ErrExceeded
	}
	return nil
}

func (l *Ledger) add(cost float64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.rolloverLocked()
	l.daily += cost
	l.total += cost
}

func (l *Ledger) rolloverLocked() {
	today := utcDate(l.now())
	if !today.Equal(l.dayStart) {
		l.dayStart = today
		l.daily = 0
	}
}

func utcDate(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}
