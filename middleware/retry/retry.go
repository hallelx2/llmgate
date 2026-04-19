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
		if !sleepBackoff(ctx, attempt, r.cfg.BaseDelay, r.cfg.MaxDelay) {
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
// Unknown is retried on the "fail open" principle — if we can't classify
// it, treat it as potentially transient and let MaxRetries bound the damage.
func defaultRetryIf(err error) bool {
	if err == nil {
		return false
	}
	switch llmgate.Classify(err) {
	case llmgate.ErrClassAuth, llmgate.ErrClassBadRequest, llmgate.ErrClassContextLength, llmgate.ErrClassContent, llmgate.ErrClassCanceled:
		return false
	default:
		return true
	}
}

func sleepBackoff(ctx context.Context, attempt int, base, max time.Duration) bool {
	d := base << attempt
	if d > max {
		d = max
	}
	jitter := time.Duration(rand.Int63n(int64(d / 2)))
	select {
	case <-time.After(d + jitter):
		return true
	case <-ctx.Done():
		return false
	}
}
