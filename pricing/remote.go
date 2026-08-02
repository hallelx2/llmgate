package pricing

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// A Source supplies model rates from somewhere outside the binary.
//
// No LLM vendor publishes machine-readable pricing, so the practical
// options are community aggregates. Two are implemented here; callers can
// add their own (an internal rate card, a billing system) by satisfying
// this interface.
type Source interface {
	// Name identifies the source in errors and diagnostics.
	Name() string
	// Fetch returns rates keyed by model ID. Keys are normalized through
	// Canonical before use, so vendor-prefixed IDs are fine.
	Fetch(ctx context.Context, c *http.Client) (map[string]Price, error)
}

// snapshot is one immutable set of remote rates plus its vintage.
type snapshot struct {
	prices map[string]Price
	asOf   time.Time
	source string
}

// RemoteConfig configures UseRemote.
type RemoteConfig struct {
	// Sources are consulted in order; the first to return usable data
	// wins. Empty means LiteLLM then OpenRouter.
	Sources []Source

	// RefreshInterval is how often to re-fetch. Zero means 24h; a
	// negative value fetches once and never refreshes.
	RefreshInterval time.Duration

	// Timeout bounds a single fetch. Zero means 30s.
	Timeout time.Duration

	// CacheDir persists the last good snapshot so a restart starts warm
	// and an offline start is not stuck on the embedded table. Empty
	// disables persistence.
	CacheDir string

	// HTTPClient overrides the client used for fetches.
	HTTPClient *http.Client

	// OnError is called for every failed fetch. Failures are otherwise
	// silent by design — a price refresh must never break a completion —
	// so wire this if you want to know the book has gone stale.
	OnError func(source string, err error)

	// OnRefresh is called after each successful refresh, with the number
	// of models loaded and the source that supplied them.
	OnRefresh func(source string, models int)
}

// UseRemote installs a refreshing remote price layer beneath any
// Register overrides and above the embedded table.
//
// It is opt-in and must stay that way: importing this package performs no
// network I/O, and a library that phoned home on init would be
// unacceptable in a dependency. Nothing here can fail a completion —
// lookups never block on the network, a failed fetch keeps the previous
// snapshot, and an empty book falls through to the compiled-in defaults.
//
// The returned stop function ends the refresh loop and removes the remote
// layer, so lookups fall back to Register overrides and the embedded
// table. Calling it more than once is safe.
func UseRemote(ctx context.Context, cfg RemoteConfig) (stop func(), err error) {
	if len(cfg.Sources) == 0 {
		cfg.Sources = []Source{LiteLLMSource{}, OpenRouterSource{}}
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 30 * time.Second
	}
	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: cfg.Timeout}
	}

	// Start warm from disk if we can, so the first lookups after a
	// restart are not stuck on the embedded table while the fetch runs.
	if cfg.CacheDir != "" {
		if snap, e := loadCache(cfg.CacheDir); e == nil {
			remote.Store(snap)
		}
	}

	refresh := func() {
		fetchCtx, cancel := context.WithTimeout(ctx, cfg.Timeout)
		defer cancel()

		for _, src := range cfg.Sources {
			raw, e := src.Fetch(fetchCtx, client)
			if e != nil {
				if cfg.OnError != nil {
					cfg.OnError(src.Name(), e)
				}
				continue
			}
			cleaned, e := vet(raw)
			if e != nil {
				if cfg.OnError != nil {
					cfg.OnError(src.Name(), e)
				}
				continue
			}

			snap := &snapshot{prices: cleaned, asOf: time.Now(), source: src.Name()}
			remote.Store(snap)
			if cfg.CacheDir != "" {
				if e := saveCache(cfg.CacheDir, snap); e != nil && cfg.OnError != nil {
					cfg.OnError(src.Name(), fmt.Errorf("persist snapshot: %w", e))
				}
			}
			if cfg.OnRefresh != nil {
				cfg.OnRefresh(src.Name(), len(cleaned))
			}
			return
		}
	}

	refresh()

	if cfg.RefreshInterval < 0 {
		var once sync.Once
		return func() { once.Do(func() { remote.Store(nil) }) }, nil
	}
	interval := cfg.RefreshInterval
	if interval == 0 {
		interval = 24 * time.Hour
	}

	done := make(chan struct{})
	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-t.C:
				refresh()
			case <-done:
				return
			case <-ctx.Done():
				return
			}
		}
	}()

	var once sync.Once
	return func() {
		once.Do(func() {
			close(done)
			remote.Store(nil)
		})
	}, nil
}

// AsOf reports when the installed remote snapshot was fetched, and which
// source supplied it. The zero time means no remote layer is active and
// every price is coming from the embedded table.
func AsOf() (t time.Time, source string) {
	if snap := remote.Load(); snap != nil {
		return snap.asOf, snap.source
	}
	return time.Time{}, ""
}

// maxDrift is how far a remote rate may move from the embedded rate
// before the whole snapshot is treated as corrupt.
//
// Real price changes are cuts of 2-5x at the extreme. An order of
// magnitude beyond that is a units error — per-token read as per-Mtok, or
// a schema change upstream — and silently adopting it would misreport
// spend by 1000x in either direction.
const maxDrift = 10.0

// vet normalizes and sanity-checks a fetched table.
//
// Anything non-positive is dropped rather than trusted: a zero rate would
// quietly report paid calls as free, which is the exact failure this
// package exists to prevent.
func vet(raw map[string]Price) (map[string]Price, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("source returned no models")
	}

	out := make(map[string]Price, len(raw))
	for id, p := range raw {
		if p.InputPerMTok <= 0 || p.OutputPerMTok <= 0 {
			continue
		}
		key := Canonical(id)
		if key == "" {
			continue
		}
		// Prefer whichever entry carries cache rates when a vendor
		// prefix collapses two IDs onto the same canonical key.
		if prev, ok := out[key]; ok && prev.CacheReadPerMTok > 0 && p.CacheReadPerMTok == 0 {
			continue
		}
		out[key] = p
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("source returned %d models, none usable", len(raw))
	}

	// Cross-check against the models we ship rates for. If a well-known
	// rate has moved by more than an order of magnitude the feed is
	// wrong, not the vendor.
	for id, known := range defaultPrices {
		got, ok := out[id]
		if !ok {
			continue
		}
		if drift(known.InputPerMTok, got.InputPerMTok) > maxDrift ||
			drift(known.OutputPerMTok, got.OutputPerMTok) > maxDrift {
			return nil, fmt.Errorf(
				"implausible rate for %s: embedded %.4f/%.4f vs remote %.4f/%.4f — treating snapshot as corrupt",
				id, known.InputPerMTok, known.OutputPerMTok, got.InputPerMTok, got.OutputPerMTok)
		}
	}

	return out, nil
}

// drift returns the ratio between two rates, always >= 1.
func drift(a, b float64) float64 {
	if a <= 0 || b <= 0 {
		return 1
	}
	return math.Max(a/b, b/a)
}

// cacheFile is the on-disk snapshot layout.
type cacheFile struct {
	AsOf   time.Time        `json:"as_of"`
	Source string           `json:"source"`
	Prices map[string]Price `json:"prices"`
}

const cacheName = "llmgate-prices.json"

// cacheTTL bounds how stale a persisted snapshot may be before it is
// ignored on startup. A week-old cache still beats the embedded table;
// a year-old one does not.
const cacheTTL = 30 * 24 * time.Hour

func loadCache(dir string) (*snapshot, error) {
	b, err := os.ReadFile(filepath.Join(dir, cacheName))
	if err != nil {
		return nil, err
	}
	var cf cacheFile
	if err := json.Unmarshal(b, &cf); err != nil {
		return nil, err
	}
	if len(cf.Prices) == 0 {
		return nil, fmt.Errorf("cached snapshot is empty")
	}
	if time.Since(cf.AsOf) > cacheTTL {
		return nil, fmt.Errorf("cached snapshot from %s is too old", cf.AsOf.Format(time.RFC3339))
	}
	return &snapshot{prices: cf.Prices, asOf: cf.AsOf, source: cf.Source + " (cached)"}, nil
}

func saveCache(dir string, snap *snapshot) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	b, err := json.Marshal(cacheFile{AsOf: snap.asOf, Source: snap.source, Prices: snap.prices})
	if err != nil {
		return err
	}
	// Write-then-rename so a crash mid-write cannot leave a truncated
	// file that fails to parse on the next start.
	tmp := filepath.Join(dir, cacheName+".tmp")
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(dir, cacheName))
}
