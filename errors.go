package llmgate

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
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
	// ErrClassGateway means the transport succeeded but a gateway in
	// front of the model reported failure in the response body — often
	// with HTTP 200. Not retryable: it is a routing or configuration
	// fault, and repeating the call reproduces it.
	ErrClassGateway
)

// String returns a human-readable label.
func (c ErrorClass) String() string {
	switch c {
	case ErrClassTransient:
		return "transient"
	case ErrClassRateLimited:
		return "rate_limited"
	case ErrClassAuth:
		return "auth"
	case ErrClassBadRequest:
		return "bad_request"
	case ErrClassContextLength:
		return "context_length"
	case ErrClassContent:
		return "content"
	case ErrClassTimeout:
		return "timeout"
	case ErrClassCanceled:
		return "canceled"
	case ErrClassGateway:
		return "gateway"
	default:
		return "unknown"
	}
}

// LLMError is a structured error returned by the adapter when enough
// information is available to classify the failure without string
// matching. Classify() checks for this type first, falling back to
// string-based heuristics only when it's not present.
//
// Provider adapters are encouraged to wrap raw errors in LLMError when
// the HTTP status code or provider-specific error code is known.
type LLMError struct {
	// Class is the pre-determined classification.
	Class ErrorClass

	// StatusCode is the HTTP status code from the provider, or 0 if
	// not applicable (e.g. network errors).
	StatusCode int

	// Provider identifies which provider returned this error.
	Provider Provider

	// Message is a human-readable description.
	Message string

	// Cause is the underlying error.
	Cause error

	// RetryAfterDur is how long the provider asked us to wait, from a
	// Retry-After header or equivalent. Zero when the provider said
	// nothing, in which case callers fall back to their own backoff.
	RetryAfterDur time.Duration
}

// Error implements the error interface.
func (e *LLMError) Error() string {
	if e.StatusCode > 0 {
		return fmt.Sprintf("%s: %d: %s", e.Provider, e.StatusCode, e.Message)
	}
	return fmt.Sprintf("%s: %s", e.Provider, e.Message)
}

// Unwrap returns the underlying cause for errors.Is / errors.As.
func (e *LLMError) Unwrap() error { return e.Cause }

// NewLLMError creates a structured LLM error. Use this in provider
// adapters when the HTTP status code is known.
func NewLLMError(provider Provider, statusCode int, message string, cause error) *LLMError {
	return &LLMError{
		Class:      classifyStatusCode(statusCode),
		StatusCode: statusCode,
		Provider:   provider,
		Message:    message,
		Cause:      cause,
	}
}

// classifyStatusCode maps an HTTP status to an ErrorClass.
func classifyStatusCode(code int) ErrorClass {
	switch {
	case code == 429:
		return ErrClassRateLimited
	case code == 401 || code == 403:
		return ErrClassAuth
	case code == 400:
		return ErrClassBadRequest
	case code == 413:
		return ErrClassContextLength
	case code >= 500:
		return ErrClassTransient
	case code == 408:
		return ErrClassTimeout
	default:
		return ErrClassUnknown
	}
}

// Classify inspects an error and returns its class. Classification
// follows this priority:
//
//  1. If err is or wraps an *LLMError, use its pre-classified Class.
//  2. Check for context sentinels (Canceled, DeadlineExceeded).
//  3. Fall back to string-based heuristics on err.Error().
//
// The string fallback is advisory — error messages from providers can
// change between releases. Prefer *LLMError for reliable classification.
func Classify(err error) ErrorClass {
	if err == nil {
		return ErrClassUnknown
	}

	// Priority 1: structured LLMError.
	var llmErr *LLMError
	if errors.As(err, &llmErr) {
		return llmErr.Class
	}

	// Priority 2: context sentinels.
	if errors.Is(err, context.Canceled) {
		return ErrClassCanceled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return ErrClassTimeout
	}

	// Priority 3: string-based heuristics (legacy fallback).
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

// RetryAfter reports how long the provider asked the caller to wait
// before retrying, if it said anything at all.
//
// Honouring it beats guessing: on a 429 the provider knows exactly when
// capacity frees up, and exponential backoff either sleeps too little and
// earns another 429 or too much and wastes wall-clock.
func RetryAfter(err error) (time.Duration, bool) {
	var llmErr *LLMError
	if errors.As(err, &llmErr) && llmErr.RetryAfterDur > 0 {
		return llmErr.RetryAfterDur, true
	}
	return 0, false
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
