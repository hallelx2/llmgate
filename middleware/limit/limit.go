// Package limit is an adaptive concurrency limiter for a provider.
//
// A fixed concurrency number is a guess about a limit the provider does
// not publish and changes without notice. Guess high and the provider
// answers 429 — which, unless every caller handles it, is a request
// silently lost; guess low and the wall clock is spent waiting on a
// semaphore that could have been wider. Measured on 2026-09-18: a chat
// provider at concurrency 6 rate-limited 13 of 21 documents into empty
// results that reported success; a System One provider under moderate
// load failed 3 of 40 batches after four minutes of retries.
//
// The limiter moves the number instead of fixing it — additive increase,
// multiplicative decrease, the way TCP finds a link's capacity:
//
//   - a slot is taken before every call and released with the outcome;
//   - a rate-limit or transport failure halves the limit at once and, if
//     the provider said how long to wait, pauses new calls until then;
//   - a window of consecutive successes widens the limit by one.
//
// It sits INSIDE retry — retry.New(cfg)(limit.Client(l)(provider)) — so
// each attempt takes a slot and the failure that triggers a retry has
// already narrowed the limit before the retry sleeps. Every change is
// reported through OnChange, because a throttled run is a fact the
// operator must be able to see.
package limit

import (
	"context"
	"sync"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/capabilities"
)

// Config tunes a Limiter. Zero values select the defaults.
type Config struct {
	// Initial is the starting limit. Default 4.
	Initial int
	// Min is the floor the limit never drops below. Default 1.
	Min int
	// Max is the ceiling the limit never rises above. Default 64.
	Max int
	// SuccessWindow is how many consecutive successes widen the limit by
	// one. Default 20 — wide enough that a lucky run does not overshoot.
	SuccessWindow int
	// Decrease is applied on failure: limit = max(Min, limit*Decrease).
	// Default 0.5.
	Decrease float64
	// OnChange, when set, is called synchronously with every limit change.
	OnChange func(Event)
	// Now overrides the clock, for tests.
	Now func() time.Time
}

// Event is one change of the limit.
type Event struct {
	// Cause is "rate_limited", "transient", or "success".
	Cause string
	// From and To are the limit before and after.
	From, To int
	// PausedFor is how long new calls are held, when the provider said
	// so; zero otherwise.
	PausedFor time.Duration
	// Err is the error that caused a decrease; nil on an increase.
	Err error
}

// Stats is a snapshot for a dashboard.
type Stats struct {
	Limit     int
	InFlight  int
	Waiting   int
	Successes int // consecutive successes toward the next increase
	// Decreases and Increases count changes since construction.
	Decreases, Increases int
	// PausedUntil is when a provider-requested pause ends; zero if none.
	PausedUntil time.Time
}

// Limiter is an adaptive semaphore. Share one per provider across every
// caller that talks to it; the point is that all of them see one limit.
type Limiter struct {
	cfg Config

	mu        sync.Mutex
	cond      *sync.Cond
	limit     int
	inflight  int
	waiting   int
	successes int
	paused    time.Time
	decs      int
	incs      int
}

// New builds a Limiter.
func New(cfg Config) *Limiter {
	if cfg.Initial <= 0 {
		cfg.Initial = 4
	}
	if cfg.Min <= 0 {
		cfg.Min = 1
	}
	if cfg.Max <= 0 {
		cfg.Max = 64
	}
	if cfg.Max < cfg.Min {
		cfg.Max = cfg.Min
	}
	if cfg.Initial < cfg.Min {
		cfg.Initial = cfg.Min
	}
	if cfg.Initial > cfg.Max {
		cfg.Initial = cfg.Max
	}
	if cfg.SuccessWindow <= 0 {
		cfg.SuccessWindow = 20
	}
	if cfg.Decrease <= 0 || cfg.Decrease >= 1 {
		cfg.Decrease = 0.5
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	l := &Limiter{cfg: cfg, limit: cfg.Initial}
	l.cond = sync.NewCond(&l.mu)
	return l
}

// Acquire blocks until a slot is free and no provider-requested pause is
// in force, or ctx ends. The returned release must be called exactly once
// with the call's outcome; nil means success.
func (l *Limiter) Acquire(ctx context.Context) (release func(err error), err error) {
	l.mu.Lock()
	l.waiting++
	// A cancelled context must wake this waiter; a paused limiter must
	// wake it when the pause ends. Both are broadcasts on the cond.
	stop := context.AfterFunc(ctx, func() {
		l.mu.Lock()
		l.cond.Broadcast()
		l.mu.Unlock()
	})
	for {
		if ctx.Err() != nil {
			l.waiting--
			l.mu.Unlock()
			stop()
			return nil, ctx.Err()
		}
		now := l.cfg.Now()
		if l.inflight < l.limit && !now.Before(l.paused) {
			break
		}
		if now.Before(l.paused) {
			// Wake when the pause ends, whatever else happens.
			time.AfterFunc(l.paused.Sub(now), func() {
				l.mu.Lock()
				l.cond.Broadcast()
				l.mu.Unlock()
			})
		}
		l.cond.Wait()
	}
	l.waiting--
	l.inflight++
	l.mu.Unlock()
	stop()

	var once sync.Once
	return func(err error) {
		once.Do(func() { l.release(err) })
	}, nil
}

func (l *Limiter) release(err error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.inflight--
	switch {
	case err == nil:
		l.successes++
		if l.successes >= l.cfg.SuccessWindow && l.limit < l.cfg.Max {
			l.successes = 0
			from := l.limit
			l.limit++
			l.incs++
			l.emit(Event{Cause: "success", From: from, To: l.limit})
		}
	case llmgate.IsRateLimited(err), llmgate.Classify(err) == llmgate.ErrClassTransient:
		cause := "transient"
		if llmgate.IsRateLimited(err) {
			cause = "rate_limited"
		}
		l.successes = 0
		from := l.limit
		to := int(float64(l.limit) * l.cfg.Decrease)
		if to < l.cfg.Min {
			to = l.cfg.Min
		}
		var pause time.Duration
		if d, ok := llmgate.RetryAfter(err); ok {
			pause = d
			until := l.cfg.Now().Add(d)
			if until.After(l.paused) {
				l.paused = until
			}
		}
		if to != from {
			l.limit = to
			l.decs++
		}
		if to != from || pause > 0 {
			l.emit(Event{Cause: cause, From: from, To: to, PausedFor: pause, Err: err})
		}
	default:
		// Auth, bad request, context length, cancellation: not about
		// capacity. The limit stands; the success streak does not.
		l.successes = 0
	}
	l.cond.Broadcast()
}

func (l *Limiter) emit(ev Event) {
	if l.cfg.OnChange != nil {
		l.cfg.OnChange(ev)
	}
}

// Limit is the current limit.
func (l *Limiter) Limit() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.limit
}

// Stats is a snapshot.
func (l *Limiter) Stats() Stats {
	l.mu.Lock()
	defer l.mu.Unlock()
	s := Stats{Limit: l.limit, InFlight: l.inflight, Waiting: l.waiting, Successes: l.successes, Decreases: l.decs, Increases: l.incs}
	if l.cfg.Now().Before(l.paused) {
		s.PausedUntil = l.paused
	}
	return s
}

// Client wraps a chat client so every Complete takes a slot.
func Client(l *Limiter) llmgate.Middleware {
	return func(inner llmgate.Client) llmgate.Client {
		return &limitedClient{inner: inner, l: l}
	}
}

type limitedClient struct {
	inner llmgate.Client
	l     *Limiter
}

func (c *limitedClient) Complete(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
	release, err := c.l.Acquire(ctx)
	if err != nil {
		return nil, err
	}
	resp, err := c.inner.Complete(ctx, req)
	release(err)
	return resp, err
}

// CountTokens is local and takes no slot.
func (c *limitedClient) CountTokens(ctx context.Context, text string) (int, error) {
	return c.inner.CountTokens(ctx, text)
}

// Capabilities delegates so Capable is not lost through wrapping.
func (c *limitedClient) Capabilities() capabilities.Capabilities { return capabilities.Of(c.inner) }

// Judge wraps a Judge so every batch takes a slot.
func Judge(l *Limiter) llmgate.JudgeMiddleware {
	return func(inner llmgate.Judge) llmgate.Judge {
		return &limitedJudge{inner: inner, l: l}
	}
}

type limitedJudge struct {
	inner llmgate.Judge
	l     *Limiter
}

func (j *limitedJudge) Judge(ctx context.Context, req llmgate.JudgeRequest) (*llmgate.Judgment, error) {
	release, err := j.l.Acquire(ctx)
	if err != nil {
		return nil, err
	}
	res, err := j.inner.Judge(ctx, req)
	release(err)
	return res, err
}
