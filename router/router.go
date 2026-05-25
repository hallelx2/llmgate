// Package router provides a Client that tries each provider in order,
// falling over between them according to a FallbackPolicy.
package router

import (
	"context"
	"errors"
	"fmt"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/capabilities"
)

// FallbackPolicy decides whether the router moves to the next client
// when the current one errors. Return true to fall over, false to surface.
type FallbackPolicy func(err error) bool

// OnRateLimit falls over on 429s only.
var OnRateLimit FallbackPolicy = llmgate.IsRateLimited

// OnTransient falls over on 429s and transient/timeout errors.
var OnTransient FallbackPolicy = func(err error) bool {
	c := llmgate.Classify(err)
	return c == llmgate.ErrClassRateLimited || c == llmgate.ErrClassTransient || c == llmgate.ErrClassTimeout
}

// Config configures New.
type Config struct {
	// Clients is the ordered list of providers to try.
	Clients []llmgate.Client
	// Fallback decides when to fall over. Nil defaults to OnTransient.
	Fallback FallbackPolicy
}

// New returns a Client that tries each provider in order, falling over
// according to the policy. The first client whose call succeeds (or
// whose error the policy declines to fall over on) decides the result.
// CountTokens goes to the first client.
//
// The router does not sleep between fallovers — compose retry.New on
// each inner client if you want backoff. This keeps the middleware
// responsibilities separate.
func New(cfg Config) (llmgate.Client, error) {
	if len(cfg.Clients) == 0 {
		return nil, errors.New("llmgate/router: requires at least one client")
	}
	if cfg.Fallback == nil {
		cfg.Fallback = OnTransient
	}
	return &router{clients: cfg.Clients, fallback: cfg.Fallback}, nil
}

type router struct {
	clients  []llmgate.Client
	fallback FallbackPolicy
}

// Complete tries each configured client in order, falling over according
// to the router's policy until one succeeds or the policy declines to
// fall over.
//
// When all clients fail, the returned error is a *RouteError that wraps
// every error encountered along the way. The primary (first) error is
// always available via RouteError.Primary for debugging.
func (r *router) Complete(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
	var errs []error
	for _, c := range r.clients {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		resp, err := c.Complete(ctx, req)
		if err == nil {
			return resp, nil
		}
		errs = append(errs, err)
		if !r.fallback(err) {
			return nil, err
		}
	}
	return nil, &RouteError{Errors: errs}
}

// CountTokens counts tokens using the first (primary) client. Fallbacks are
// intentionally not consulted — token counts are meant to be deterministic.
func (r *router) CountTokens(ctx context.Context, text string) (int, error) {
	return r.clients[0].CountTokens(ctx, text)
}

// Capabilities reports the first client's capabilities. This is the right
// choice when the router is ordered by preference — callers see what the
// primary can do, and fallbacks are expected to be at least as capable.
func (r *router) Capabilities() capabilities.Capabilities { return capabilities.Of(r.clients[0]) }

// RouteError is returned when all configured providers failed. It wraps
// every error from each provider so callers can inspect the full failure
// chain, not just the last attempt.
type RouteError struct {
	// Errors contains one error per attempted provider, in order.
	Errors []error
}

// Primary returns the error from the first (preferred) provider. This is
// usually the most actionable one for debugging — e.g. if the primary
// returned 401 but the fallback returned 503, you want to see the 401.
func (e *RouteError) Primary() error {
	if len(e.Errors) == 0 {
		return nil
	}
	return e.Errors[0]
}

// Last returns the error from the final provider attempted.
func (e *RouteError) Last() error {
	if len(e.Errors) == 0 {
		return nil
	}
	return e.Errors[len(e.Errors)-1]
}

// Error returns a human-readable summary listing all provider failures.
func (e *RouteError) Error() string {
	if len(e.Errors) == 0 {
		return "llmgate/router: all providers failed (no errors recorded)"
	}
	if len(e.Errors) == 1 {
		return e.Errors[0].Error()
	}
	return fmt.Sprintf("llmgate/router: all %d providers failed: primary=%v, last=%v",
		len(e.Errors), e.Errors[0], e.Errors[len(e.Errors)-1])
}

// Unwrap returns the primary error so errors.Is/As work against it.
func (e *RouteError) Unwrap() error {
	return e.Primary()
}
