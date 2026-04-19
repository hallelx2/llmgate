package llmgate

import (
	"context"
	"errors"
	"strings"
)

// ErrorClass categorizes provider errors for routing and retry decisions.
type ErrorClass int

const (
	// ErrClassUnknown is the default when no pattern matches.
	ErrClassUnknown ErrorClass = iota
	// ErrClassTransient covers network blips and 5xx server errors. Retry.
	ErrClassTransient
	// ErrClassRateLimited is a 429. Retry with backoff or fall over.
	ErrClassRateLimited
	// ErrClassAuth is 401/403. Do not retry.
	ErrClassAuth
	// ErrClassBadRequest is 400 / malformed input. Do not retry.
	ErrClassBadRequest
	// ErrClassContextLength means the request exceeded the model's context window.
	ErrClassContextLength
	// ErrClassContent is a content-policy / safety refusal.
	ErrClassContent
	// ErrClassTimeout is a context timeout or provider-side timeout.
	ErrClassTimeout
	// ErrClassCanceled is context.Canceled.
	ErrClassCanceled
)

// Classify inspects an error (including wrapped langchaingo / googleapi /
// net errors) and returns its class. Classification is advisory, based on
// string matching + errors.Is on context sentinels.
func Classify(err error) ErrorClass {
	if err == nil {
		return ErrClassUnknown
	}
	if errors.Is(err, context.Canceled) {
		return ErrClassCanceled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return ErrClassTimeout
	}
	s := strings.ToLower(err.Error())

	// Rate limit — check before generic 4xx matchers.
	if strings.Contains(s, "429") || strings.Contains(s, "rate limit") || strings.Contains(s, "rate_limit") || strings.Contains(s, "too many requests") || strings.Contains(s, "quota") {
		return ErrClassRateLimited
	}

	// Auth.
	if strings.Contains(s, "401") || strings.Contains(s, "403") ||
		strings.Contains(s, "unauthorized") || strings.Contains(s, "forbidden") ||
		strings.Contains(s, "authentication") || strings.Contains(s, "invalid_api_key") ||
		strings.Contains(s, "invalid api key") || strings.Contains(s, "permission denied") {
		return ErrClassAuth
	}

	// Context length.
	if strings.Contains(s, "context length") || strings.Contains(s, "maximum context") ||
		strings.Contains(s, "context_length_exceeded") || strings.Contains(s, "too long") ||
		(strings.Contains(s, "max_tokens") && strings.Contains(s, "exceed")) {
		return ErrClassContextLength
	}

	// Content policy.
	if strings.Contains(s, "content filter") || strings.Contains(s, "content_filter") ||
		strings.Contains(s, "safety") || strings.Contains(s, "blocked") ||
		strings.Contains(s, "content policy") || strings.Contains(s, "content_policy") {
		return ErrClassContent
	}

	// Timeout.
	if strings.Contains(s, "timeout") || strings.Contains(s, "timed out") || strings.Contains(s, "deadline exceeded") {
		return ErrClassTimeout
	}

	// Transient: 5xx, network drops.
	if strings.Contains(s, "500") || strings.Contains(s, "502") || strings.Contains(s, "503") || strings.Contains(s, "504") ||
		strings.Contains(s, "bad gateway") || strings.Contains(s, "service unavailable") ||
		strings.Contains(s, "gateway timeout") || strings.Contains(s, "eof") ||
		strings.Contains(s, "connection reset") || strings.Contains(s, "connection refused") ||
		strings.Contains(s, "broken pipe") || strings.Contains(s, "no such host") ||
		strings.Contains(s, "transient") || strings.Contains(s, "temporarily") ||
		strings.Contains(s, "try again") || strings.Contains(s, "overloaded") {
		return ErrClassTransient
	}

	// Generic bad request — check after more specific 4xx cases.
	if strings.Contains(s, "400") || strings.Contains(s, "bad request") || strings.Contains(s, "invalid_request") || strings.Contains(s, "invalid request") {
		return ErrClassBadRequest
	}

	return ErrClassUnknown
}

// IsRateLimited reports whether err is a rate-limit error.
func IsRateLimited(err error) bool { return Classify(err) == ErrClassRateLimited }

// IsTransient reports whether err is a transient (retryable) error,
// including timeouts and rate limits.
func IsTransient(err error) bool {
	c := Classify(err)
	return c == ErrClassTransient || c == ErrClassTimeout || c == ErrClassRateLimited
}

// IsAuth reports whether err is an authentication / authorization error.
func IsAuth(err error) bool { return Classify(err) == ErrClassAuth }
