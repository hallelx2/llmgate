package retry_test

import (
	"context"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/middleware/retry"
)

// failing returns the same error until it has been called n times.
type failing struct {
	err       error
	failFirst int
	calls     int
}

func (f *failing) Complete(context.Context, llmgate.Request) (*llmgate.Response, error) {
	f.calls++
	if f.calls <= f.failFirst {
		return nil, f.err
	}
	return &llmgate.Response{Content: "ok"}, nil
}

func (f *failing) CountTokens(context.Context, string) (int, error) { return 0, nil }

// TestNoPanicOnExtremeConfig covers the two ways sleepBackoff could
// panic, both reachable from a plausible config.
//
// A large MaxRetries overflowed base<<attempt into a negative duration,
// which slipped past the `> max` clamp and reached rand.Int63n with a
// non-positive bound. A sub-2ns BaseDelay made the jitter bound zero,
// which panics the same way.
func TestNoPanicOnExtremeConfig(t *testing.T) {
	tests := []struct {
		name       string
		maxRetries int
		base, max  time.Duration
	}{
		{"overflowing attempt count", 40, 500 * time.Millisecond, time.Microsecond},
		{"sub-nanosecond base", 5, 1, time.Microsecond},
		{"zero-ish jitter window", 3, 2, 3},
		{"huge base", 4, time.Duration(1) << 61, time.Millisecond},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := retry.New(retry.Config{
				MaxRetries: tc.maxRetries,
				BaseDelay:  tc.base,
				MaxDelay:   tc.max,
			})(&failing{err: errTransient, failFirst: 1000})

			// The assertion is simply that this returns rather than
			// panicking; the error itself is expected.
			if _, err := c.Complete(context.Background(), llmgate.Request{}); err == nil {
				t.Fatal("expected the call to fail after exhausting retries")
			}
		})
	}
}

var errTransient = &llmgate.LLMError{
	Class:    llmgate.ErrClassTransient,
	Provider: llmgate.ProviderAnthropic,
	Message:  "503 service unavailable",
}

// TestRetryAfterHonoured: when the provider says how long to wait, that
// beats guessing. A 60ms Retry-After against a 1ns base means the call
// cannot possibly return before ~60ms.
func TestRetryAfterHonoured(t *testing.T) {
	rateLimited := &llmgate.LLMError{
		Class:         llmgate.ErrClassRateLimited,
		StatusCode:    429,
		Provider:      llmgate.ProviderAnthropic,
		Message:       "rate limited",
		RetryAfterDur: 60 * time.Millisecond,
	}

	c := retry.New(retry.Config{
		MaxRetries: 1,
		BaseDelay:  time.Nanosecond, // would otherwise retry ~instantly
		MaxDelay:   time.Second,
	})(&failing{err: rateLimited, failFirst: 1})

	start := time.Now()
	if _, err := c.Complete(context.Background(), llmgate.Request{}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if elapsed := time.Since(start); elapsed < 55*time.Millisecond {
		t.Errorf("retried after %v, want at least the provider's 60ms Retry-After", elapsed)
	}
}

// TestRetryAfterClampedToMaxDelay: honouring the header must not let a
// mistaken or hostile value park the caller indefinitely.
func TestRetryAfterClampedToMaxDelay(t *testing.T) {
	absurd := &llmgate.LLMError{
		Class:         llmgate.ErrClassRateLimited,
		Provider:      llmgate.ProviderOpenAI,
		Message:       "rate limited",
		RetryAfterDur: time.Hour,
	}

	c := retry.New(retry.Config{
		MaxRetries: 1,
		BaseDelay:  time.Millisecond,
		MaxDelay:   50 * time.Millisecond,
	})(&failing{err: absurd, failFirst: 1})

	start := time.Now()
	if _, err := c.Complete(context.Background(), llmgate.Request{}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("waited %v — a one-hour Retry-After must clamp to MaxDelay", elapsed)
	}
}

// TestGatewayErrorNotRetried: a gateway reporting failure in a 200 body
// is a configuration fault. Retrying it burns four paid attempts on
// something that cannot succeed.
func TestGatewayErrorNotRetried(t *testing.T) {
	gatewayErr := &llmgate.LLMError{
		Class:    llmgate.ErrClassGateway,
		Provider: llmgate.ProviderAnthropic,
		Message:  `empty response — {"code":500,"msg":"404 NOT_FOUND"}`,
	}

	inner := &failing{err: gatewayErr, failFirst: 1000}
	c := retry.New(retry.Config{MaxRetries: 3, BaseDelay: time.Nanosecond})(inner)

	if _, err := c.Complete(context.Background(), llmgate.Request{}); err == nil {
		t.Fatal("expected the gateway error to surface")
	}
	if inner.calls != 1 {
		t.Errorf("inner called %d times, want 1 — a gateway fault is permanent", inner.calls)
	}
}
