package pricing

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
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
			cleaned, dropped, e := vet(raw)
			if e != nil {
				if cfg.OnError != nil {
					cfg.OnError(src.Name(), e)
				}
				continue
			}
			// Dropped rows are not fatal, but they are evidence that a
			// feed has gone wrong somewhere. Report them rather than
			// letting the snapshot look clean.
			if len(dropped) > 0 && cfg.OnError != nil {
				cfg.OnError(src.Name(), fmt.Errorf(
					"dropped %d entries with implausible rates: %s",
					len(dropped), strings.Join(sortedSample(dropped, 5), ", ")))
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
func vet(raw map[string]Price) (map[string]Price, []string, error) {
	if len(raw) == 0 {
		return nil, nil, fmt.Errorf("source returned no models")
	}

	// Drop implausible entries before the collapse, not after. Doing it
	// afterwards let one poisoned row decide a canonical key and then
	// condemn the whole snapshot: LiteLLM carries
	// "wandb/zai-org/GLM-4.5" at 55000/200000, which canonicalizes onto
	// glm-4.5 and reads as a 91000x move against the embedded rate. One
	// bad row upstream must cost us that row, not the refresh.
	var dropped []string
	usable := make(map[string]Price, len(raw))
	for id, p := range raw {
		if p.InputPerMTok <= 0 || p.OutputPerMTok <= 0 {
			continue
		}
		if Canonical(id) == "" {
			continue
		}
		if known, ok := defaultPrices[Canonical(id)]; ok {
			if drift(known.InputPerMTok, p.InputPerMTok) > maxDrift ||
				drift(known.OutputPerMTok, p.OutputPerMTok) > maxDrift {
				dropped = append(dropped, id)
				continue
			}
		}
		usable[id] = p
	}

	if len(usable) == 0 {
		return nil, dropped, fmt.Errorf("source returned %d models, none usable", len(raw))
	}

	// A handful of bad rows is upstream noise. A large fraction of them is
	// a units error or a schema change, and adopting it would misreport
	// spend by orders of magnitude — so that still condemns the snapshot.
	if len(dropped) > len(raw)/10 {
		return nil, dropped, fmt.Errorf(
			"%d of %d entries have implausible rates — treating snapshot as corrupt",
			len(dropped), len(raw))
	}

	// Collapse to canonical keys. Many upstream IDs land on the same key
	// with genuinely different prices, one per vendor selling the model,
	// so the winner is chosen by rank (see provider.go) rather than by
	// whichever the map happened to yield last.
	out := make(map[string]Price, len(usable))
	winner := make(map[string]string, len(usable))
	for id, p := range usable {
		key := Canonical(id)
		if prevID, ok := winner[key]; ok && !better(id, p, prevID, out[key]) {
			continue
		}
		out[key] = p
		winner[key] = id
	}

	// A feed that omits a cache rate is silent about it, not asserting
	// zero. Where the embedded table has a published rate for the same
	// model, keep it: "zai/glm-4.6" wins its collision on rank but carries
	// no cache-write rate, and without this the read rate would fall
	// through to the family multiplier — 0.30 against the 0.11 z.ai
	// actually charges, a 2.7x error on every cached token.
	for key, p := range out {
		known, ok := defaultPrices[key]
		if !ok {
			continue
		}
		if p.CacheReadPerMTok == 0 && known.CacheReadPerMTok > 0 {
			p.CacheReadPerMTok = known.CacheReadPerMTok
		}
		if p.CacheWritePerMTok == 0 && known.CacheWritePerMTok > 0 {
			p.CacheWritePerMTok = known.CacheWritePerMTok
		}
		if p.ReasoningPerMTok == 0 && known.ReasoningPerMTok > 0 {
			p.ReasoningPerMTok = known.ReasoningPerMTok
		}
		out[key] = p
	}

	return out, dropped, nil
}

// sortedSample returns up to n of the given IDs in sorted order, so an
// error message naming "some of" a list names the same ones every time.
func sortedSample(ids []string, n int) []string {
	s := slices.Clone(ids)
	slices.Sort(s)
	if len(s) > n {
		s = append(s[:n:n], fmt.Sprintf("and %d more", len(s)-n))
	}
	return s
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
