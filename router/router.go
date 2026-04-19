// Package router provides a Client that tries each provider in order,
// falling over between them according to a FallbackPolicy.
package router

import (
	"context"
	"errors"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/capabilities"
)

// FallbackPolicy decides whether the router moves to the next client
// when the current one errors. Return true to fall over, false to surface.
type FallbackPolicy func(err error) bool

// OnRateLimit falls over on 429s only.
var OnRateLimit FallbackPolicy = func(err error) bool { return llmgate.IsRateLimited(err) }

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

func (r *router) Complete(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
	var lastErr error
	for _, c := range r.clients {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		resp, err := c.Complete(ctx, req)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		if !r.fallback(err) {
			return nil, err
		}
	}
	return nil, lastErr
}

func (r *router) CountTokens(ctx context.Context, text string) (int, error) {
	return r.clients[0].CountTokens(ctx, text)
}

// Capabilities reports the first client's capabilities. This is the right
// choice when the router is ordered by preference — callers see what the
// primary can do, and fallbacks are expected to be at least as capable.
func (r *router) Capabilities() capabilities.Capabilities { return capabilities.Of(r.clients[0]) }
