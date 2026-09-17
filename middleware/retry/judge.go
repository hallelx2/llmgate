package retry

import (
	"context"

	"github.com/hallelx2/llmgate"
)

// NewJudge returns a JudgeMiddleware that retries Judge on transient
// errors, using the same backoff, jitter, and Retry-After handling as the
// Client middleware.
//
// It shares Config and every helper with New deliberately. The backoff
// logic here is not as simple as it looks — it carries guards against an
// overflowing shift and a sub-2ns base delay, both of which panic in
// rand.Int63n on a non-positive bound (see expBackoff and jitterFor, and
// TestNoPanicOnExtremeConfig). A second implementation for the judgment
// seam would have to rediscover all of that, so there isn't one.
//
// Retry matters more here than the request count suggests. A System One
// call carries a whole batch of questions, so a dropped call loses every
// answer in it, and the provider's published rate limits are explicitly
// documented as changing without notice — which makes a 429 an ordinary
// event to absorb rather than an exceptional one to surface.
func NewJudge(cfg Config) llmgate.JudgeMiddleware {
	cfg = withDefaults(cfg)
	return func(inner llmgate.Judge) llmgate.Judge {
		return &retryJudge{inner: inner, cfg: cfg}
	}
}

type retryJudge struct {
	inner llmgate.Judge
	cfg   Config
}

// Judge invokes the inner judge with exponential backoff, retrying only
// errors the configured predicate accepts and honouring ctx cancellation.
func (r *retryJudge) Judge(ctx context.Context, req llmgate.JudgeRequest) (*llmgate.Judgment, error) {
	var lastErr error
	for attempt := 0; attempt <= r.cfg.MaxRetries; attempt++ {
		j, err := r.inner.Judge(ctx, req)
		if err == nil {
			return j, nil
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
