package llmgate_test

import (
	"context"
	"errors"
	"testing"

	"github.com/hallelx2/llmgate"
)

func TestClassify(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want llmgate.ErrorClass
	}{
		{"nil", nil, llmgate.ErrClassUnknown},
		{"rate limit 429", errors.New("HTTP 429: too many requests"), llmgate.ErrClassRateLimited},
		{"rate limit text", errors.New("rate limit exceeded"), llmgate.ErrClassRateLimited},
		{"auth 401", errors.New("HTTP 401 unauthorized"), llmgate.ErrClassAuth},
		{"invalid api key", errors.New("invalid_api_key provided"), llmgate.ErrClassAuth},
		{"bad request", errors.New("HTTP 400 bad request"), llmgate.ErrClassBadRequest},
		{"context length", errors.New("context length exceeded"), llmgate.ErrClassContextLength},
		{"content filter", errors.New("response blocked by safety filter"), llmgate.ErrClassContent},
		{"timeout", errors.New("request timeout"), llmgate.ErrClassTimeout},
		{"500", errors.New("HTTP 500 internal server error"), llmgate.ErrClassTransient},
		{"bad gateway", errors.New("bad gateway"), llmgate.ErrClassTransient},
		{"ctx canceled", context.Canceled, llmgate.ErrClassCanceled},
		{"ctx deadline", context.DeadlineExceeded, llmgate.ErrClassTimeout},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := llmgate.Classify(c.err); got != c.want {
				t.Fatalf("Classify(%v) = %d, want %d", c.err, got, c.want)
			}
		})
	}
}

func TestPredicates(t *testing.T) {
	if !llmgate.IsRateLimited(errors.New("429 rate limit")) {
		t.Fatalf("IsRateLimited should match 429")
	}
	if llmgate.IsAuth(errors.New("429")) {
		t.Fatalf("IsAuth should not match 429")
	}
	if !llmgate.IsAuth(errors.New("401 unauthorized")) {
		t.Fatalf("IsAuth should match 401")
	}
	if !llmgate.IsTransient(errors.New("503 service unavailable")) {
		t.Fatalf("IsTransient should match 503")
	}
}
