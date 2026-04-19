// Package cache provides an in-memory LRU response cache middleware
// for llmgate.Client.
package cache

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"sync"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/capabilities"
)

// Config configures New.
type Config struct {
	// Capacity is the maximum number of cached entries; <=0 defaults to 256.
	Capacity int
	// TTL is the per-entry expiry; 0 means no expiry.
	TTL time.Duration
	// Now overrides time.Now for tests; nil uses time.Now.
	Now func() time.Time
}

// New returns a Middleware that caches Response by request key.
// Errors are never cached. CountTokens is not cached. Cached responses
// have Usage.CostUSD zeroed out and FromCache set to true so callers
// can see cache savings.
func New(cfg Config) llmgate.Middleware {
	if cfg.Capacity <= 0 {
		cfg.Capacity = 256
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	c := &lruCache{cap: cfg.Capacity, ttl: cfg.TTL, now: now, ll: list.New(), m: map[string]*list.Element{}}
	return func(inner llmgate.Client) llmgate.Client {
		return &cacheClient{inner: inner, c: c}
	}
}

type cacheClient struct {
	inner llmgate.Client
	c     *lruCache
}

// Complete returns a cached response when one is available for the request
// shape; on a miss it forwards to the inner client and stores the result.
func (c *cacheClient) Complete(ctx context.Context, req llmgate.Request) (*llmgate.Response, error) {
	key := cacheKey(req)
	if resp, ok := c.c.get(key); ok {
		clone := *resp
		clone.FromCache = true
		clone.Usage.CostUSD = 0
		return &clone, nil
	}
	resp, err := c.inner.Complete(ctx, req)
	if err != nil {
		return nil, err
	}
	// Store a copy so later mutations by callers don't poison the cache.
	stored := *resp
	c.c.put(key, &stored)
	return resp, nil
}

// CountTokens passes through to the inner client; counts are not cached.
func (c *cacheClient) CountTokens(ctx context.Context, text string) (int, error) {
	return c.inner.CountTokens(ctx, text)
}

// Capabilities delegates to the inner client.
func (c *cacheClient) Capabilities() capabilities.Capabilities { return capabilities.Of(c.inner) }

// cacheKey hashes the request fields that change the response.
func cacheKey(req llmgate.Request) string {
	h := sha256.New()
	h.Write([]byte(req.Model))
	h.Write([]byte{0})
	for _, m := range req.Messages {
		h.Write([]byte(m.Role))
		h.Write([]byte{0})
		h.Write([]byte(m.Content))
		h.Write([]byte{0})
	}
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(req.MaxTokens))
	h.Write(buf[:])
	binary.LittleEndian.PutUint64(buf[:], math.Float64bits(req.Temperature))
	h.Write(buf[:])
	if req.JSONMode {
		h.Write([]byte{1})
	} else {
		h.Write([]byte{0})
	}
	h.Write(req.JSONSchema)
	h.Write([]byte{0})
	for _, t := range req.Tools {
		h.Write([]byte(t.Name))
		h.Write([]byte{0})
		h.Write([]byte(t.Description))
		h.Write([]byte{0})
		h.Write(t.InputSchema)
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

type lruCache struct {
	cap int
	ttl time.Duration
	now func() time.Time

	mu sync.Mutex
	ll *list.List
	m  map[string]*list.Element
}

type lruEntry struct {
	key     string
	val     *llmgate.Response
	expires time.Time // zero = never
}

func (c *lruCache) get(key string) (*llmgate.Response, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.m[key]
	if !ok {
		return nil, false
	}
	en := e.Value.(*lruEntry)
	if !en.expires.IsZero() && c.now().After(en.expires) {
		c.ll.Remove(e)
		delete(c.m, key)
		return nil, false
	}
	c.ll.MoveToFront(e)
	return en.val, true
}

func (c *lruCache) put(key string, val *llmgate.Response) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.m[key]; ok {
		e.Value.(*lruEntry).val = val
		if c.ttl > 0 {
			e.Value.(*lruEntry).expires = c.now().Add(c.ttl)
		}
		c.ll.MoveToFront(e)
		return
	}
	en := &lruEntry{key: key, val: val}
	if c.ttl > 0 {
		en.expires = c.now().Add(c.ttl)
	}
	e := c.ll.PushFront(en)
	c.m[key] = e
	for c.ll.Len() > c.cap {
		old := c.ll.Back()
		if old == nil {
			break
		}
		c.ll.Remove(old)
		delete(c.m, old.Value.(*lruEntry).key)
	}
}
