// Package retry provides a retry middleware with exponential backoff +
// jitter for llmgate.Client.
package retry

import (
	"context"
	"math/rand"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/capabilities"
)

// Config tunes New.
type Config struct {
	// MaxRetries is the number of additional attempts after the first call.
	// Zero falls back to 3.
	MaxRetries int

	// BaseDelay is the starting backoff; doubled each attempt up to MaxDelay.
	// Zero falls back to 500ms.
	BaseDelay time.Duration

	// MaxDelay caps the per-attempt sleep.
	// Zero falls back to 10s.
	MaxDelay time.Duration

	// RetryIf decides whether an error is worth retrying. If nil, any
	// non-context error is retried — which is close enough for most
	// provider errors (429 / 5xx / transient network) to get caught.
	//
	// A richer version lives in the roadmap (provider-specific error
	// classification once langchaingo exposes structured errors).
	RetryIf func(err error) bool
}

// New returns a Middleware that retries Complete on transient errors
// with exponential backoff + jitter.
func New(cfg Config) llmgate.Middleware {
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 3
	}
	if cfg.BaseDelay <= 0 {
		cfg.BaseDelay = 500 * time.Millisecond
	}
	if cfg.MaxDelay <= 0 {
		cfg.MaxDelay = 10 * time.Second
	}
	if cfg.RetryIf == nil {
		cfg.RetryIf = defaultRetryIf
	}
	return func(inner llmgate.Client) llmgate.Client {
		return &retryClient{inner: inner, cfg: cfg}
	}
}

type retryClient struct {
	inner llmgate.Client
	cfg   Config
}

// Complete invokes the inner client with exponential backoff, retrying only
// errors that the configured predicate accepts and honouring ctx cancellation.
func (r *retryClient) Complete(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
	var lastErr error
	for attempt := 0; attempt <= r.cfg.MaxRetries; attempt++ {
		resp, err := r.inner.Complete(ctx, req)
		if err == nil {
			return resp, nil
		}
		lastErr = err

		// Context cancellations are never retried.
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if !r.cfg.RetryIf(err) {
			return nil, err
		}
		if attempt == r.cfg.MaxRetries {
			break
		}
		if !sleepBackoff(ctx, err, attempt, r.cfg.BaseDelay, r.cfg.MaxDelay) {
			return nil, ctx.Err()
		}
	}
	return nil, lastErr
}

// CountTokens passes through to the inner client without retry.
func (r *retryClient) CountTokens(ctx context.Context, text string) (int, error) {
	// No retry on counting; it's advisory and callers fall back to estimates.
	return r.inner.CountTokens(ctx, text)
}

// Capabilities delegates to the inner client so Capable isn't lost through
// middleware wrapping.
func (r *retryClient) Capabilities() capabilities.Capabilities { return capabilities.Of(r.inner) }

// defaultRetryIf retries errors that Classify marks as Transient,
// RateLimited, Timeout, or Unknown. Auth, BadRequest, ContextLength,
// Content, and Canceled errors are surfaced on the first failure.
//
// IMPORTANT — fail-open principle: Unknown errors are retried. This is
// intentional. If we can't classify an error (e.g. new provider error
// format), it's safer to retry it (bounded by MaxRetries) than to
// immediately surface it and lose a potentially-successful request.
// If this behavior is wrong for your use case, provide a custom
// RetryIf function that returns false for ErrClassUnknown.
func defaultRetryIf(err error) bool {
	if err == nil {
		return false
	}
	switch llmgate.Classify(err) {
	case llmgate.ErrClassAuth, llmgate.ErrClassBadRequest, llmgate.ErrClassContextLength,
		llmgate.ErrClassContent, llmgate.ErrClassCanceled, llmgate.ErrClassGateway:
		return false
	default:
		return true
	}
}

// backoffFor computes the delay before the next attempt.
//
// A provider that sent Retry-After has told us exactly how long to wait,
// which beats any guess — but it is still clamped, so a hostile or
// mistaken header cannot park the caller for an hour.
func backoffFor(err error, attempt int, base, max time.Duration) time.Duration {
	if after, ok := llmgate.RetryAfter(err); ok && after > 0 {
		return min(after, max)
	}
	return expBackoff(attempt, base, max)
}

// expBackoff doubles base per attempt, saturating at max.
//
// The shift is guarded rather than computed directly: base<<attempt
// overflows int64 around attempt 35 for a sub-second base and goes
// negative, which slips past a `> max` clamp and then panics in
// rand.Int63n on a non-positive bound.
func expBackoff(attempt int, base, max time.Duration) time.Duration {
	d := base
	for range attempt {
		if d >= max/2 {
			return max
		}
		d *= 2
	}
	if d > max {
		d = max
	}
	return d
}

// jitterFor returns a random 0..d/2 stagger, so a fleet retrying together
// does not re-collide on the same tick.
func jitterFor(d time.Duration) time.Duration {
	half := int64(d / 2)
	if half <= 0 {
		// rand.Int63n panics on a non-positive bound, which a BaseDelay
		// under 2ns would otherwise reach.
		return 0
	}
	return time.Duration(rand.Int63n(half))
}

func sleepBackoff(ctx context.Context, err error, attempt int, base, max time.Duration) bool {
	d := backoffFor(err, attempt, base, max)
	select {
	case <-time.After(d + jitterFor(d)):
		return true
	case <-ctx.Done():
		return false
	}
}
