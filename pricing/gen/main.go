// Command gen regenerates the embedded price table in
// pricing/defaults_gen.go from the live LiteLLM feed.
//
// Run it with `go generate ./pricing`. It is not part of the library build.
//
// The embedded table exists for the cases where the remote layer cannot
// help: UseRemote not enabled, the network down, or the first refresh still
// in flight. That makes staleness a correctness problem rather than an
// untidiness one — before this generator existed, gemini-2.5-flash sat at
// an output rate of 0.60 against a real 2.50, under-reporting spend on that
// model by more than 4x for anyone who had not opted into remote pricing.
//
// Rather than reimplement rate selection, the generator installs a real
// snapshot and reads it back through pricing.Lookup. Everything the library
// does to choose between the many upstream entries for one model — vendor
// precedence over resellers, dropping implausible rows, backfilling cache
// rates — therefore applies here identically, and the embedded table cannot
// drift from the rules the remote layer follows.
package main

import (
	"context"
	"fmt"
	"go/format"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/hallelx2/llmgate/pricing"
)

// model is one curated entry. id is the key written into the generated
// table; from is the upstream ID to read it from when the two differ.
type model struct {
	id   string
	from string // optional; defaults to id
}

func m(id string) model           { return model{id: id} }
func alias(id, from string) model { return model{id: id, from: from} }

func (x model) source() string {
	if x.from != "" {
		return x.from
	}
	return x.id
}

// curated is the editorial decision this generator cannot make for itself:
// which models are worth carrying in the binary. The feed has over 2,000.
//
// The rule of thumb is models a caller might plausibly run without having
// configured remote pricing — current frontier models from the four
// families llmgate classifies, plus the GLM line vectorless runs in
// production. Ordering here drives the ordering of the generated file.
var curated = []struct {
	group  string
	models []model
}{
	{"Anthropic", []model{
		m("claude-opus-5"), m("claude-opus-4-6"), m("claude-opus-4-5"), m("claude-opus-4-1"),
		m("claude-sonnet-5"), m("claude-sonnet-4-6"), m("claude-sonnet-4-5"), m("claude-sonnet-4"),
		m("claude-haiku-4-5"),
		// Anthropic put the generation before the tier for the 3.x line
		// and after it from 4.x on. Upstream follows suit; llmgate has
		// always keyed this one the modern way, so keep that and source
		// it from the ID the feed actually carries.
		alias("claude-haiku-3-5", "claude-3-5-haiku"),
	}},
	{"OpenAI", []model{
		m("gpt-4o"), m("gpt-4o-mini"), m("gpt-4.1"), m("gpt-4.1-mini"), m("gpt-4.1-nano"),
		m("o3"), m("o3-mini"), m("o4-mini"),
	}},
	{"Google", []model{
		m("gemini-2.5-pro"), m("gemini-2.5-flash"), m("gemini-2.0-flash"),
	}},
	{"Zhipu / Z.ai GLM", []model{
		m("glm-5.1"), m("glm-5"), m("glm-4.7"), m("glm-4.6"), m("glm-4.5"), m("glm-4.5-air"),
	}},
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "gen:", err)
		os.Exit(1)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	var fetchErr error
	stop, err := pricing.UseRemote(ctx, pricing.RemoteConfig{
		Sources:         []pricing.Source{pricing.LiteLLMSource{}},
		RefreshInterval: -1,
		Timeout:         2 * time.Minute,
		OnError:         func(src string, e error) { fetchErr = fmt.Errorf("%s: %w", src, e) },
		OnRefresh: func(src string, n int) {
			fmt.Fprintf(os.Stderr, "gen: %s supplied %d models\n", src, n)
		},
	})
	if err != nil {
		return err
	}
	defer stop()

	asOf, source := pricing.AsOf()
	if asOf.IsZero() {
		if fetchErr != nil {
			return fmt.Errorf("no snapshot installed: %w", fetchErr)
		}
		return fmt.Errorf("no snapshot installed and no error reported")
	}

	// A missing model is a hard failure, not an omission: silently dropping
	// one would shrink the embedded table without anyone noticing.
	//
	// Require the exact key. Lookup would happily fall back to a prefix
	// match, which is how claude-haiku-3-5 once picked up Claude 1's
	// 8.00/24.00 — a curated ID that is not really in the feed must fail
	// loudly here, not be quietly filled with a sibling's rate.
	present := map[string]bool{}
	for _, k := range pricing.KnownModels() {
		present[k] = true
	}

	var missing []string
	prices := map[string]pricing.Price{}
	for _, g := range curated {
		for _, mo := range g.models {
			src := mo.source()
			if !present[src] {
				missing = append(missing, src)
				continue
			}
			p, ok := pricing.Lookup(src)
			if !ok {
				missing = append(missing, src)
				continue
			}
			prices[mo.id] = p
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("not present in %s: %v — remove them from the curated list "+
			"or fix the ID before regenerating", source, missing)
	}

	src, err := render(prices, asOf, source)
	if err != nil {
		return err
	}

	// Every run stamps a new fetch time, so writing unconditionally would
	// dirty the file even when no vendor moved a price — and the weekly
	// drift job, which is just `go generate` plus `git diff --exit-code`,
	// would then fail every week and teach everyone to ignore it. Rewrite
	// only when a rate actually changed.
	if old, e := os.ReadFile("defaults_gen.go"); e == nil && sameRates(old, src) {
		fmt.Fprintln(os.Stderr, "gen: rates unchanged, leaving defaults_gen.go alone")
		return nil
	}
	return os.WriteFile("defaults_gen.go", src, 0o644)
}

// sameRates compares two generated files ignoring the provenance stamps,
// which differ on every run by construction.
func sameRates(a, b []byte) bool {
	return stripStamps(a) == stripStamps(b)
}

func stripStamps(src []byte) string {
	var kept []string
	for line := range strings.SplitSeq(string(src), "\n") {
		t := strings.TrimSpace(strings.TrimSuffix(line, "\r"))
		if strings.HasPrefix(t, "// Source:") || strings.HasPrefix(t, "var defaultsAsOf") {
			continue
		}
		kept = append(kept, t)
	}
	return strings.Join(kept, "\n")
}

func render(prices map[string]pricing.Price, asOf time.Time, source string) ([]byte, error) {
	var b []byte
	add := func(format string, args ...any) {
		b = fmt.Appendf(b, format, args...)
	}

	add("// Code generated by pricing/gen. DO NOT EDIT.\n")
	add("//\n")
	add("// Source: %s, fetched %s.\n", source, asOf.UTC().Format(time.RFC3339))
	add("// Regenerate with: go generate ./pricing\n\n")
	add("package pricing\n\n")
	add("// defaultPrices is the compiled-in price table. It serves every lookup\n")
	add("// that the override and remote layers do not answer: no UseRemote, no\n")
	add("// network, or a first refresh still in flight.\n")
	add("//\n")
	add("// Rates are USD per 1,000,000 tokens. A zero cache rate means the feed\n")
	add("// published none, and cacheRates falls back to the family multiplier.\n")
	add("var defaultPrices = map[string]Price{\n")

	for _, g := range curated {
		add("\t// %s\n", g.group)
		for _, mo := range g.models {
			p := prices[mo.id]
			add("\t%q: {InputPerMTok: %s, OutputPerMTok: %s",
				mo.id, num(p.InputPerMTok), num(p.OutputPerMTok))
			if p.CacheWritePerMTok > 0 {
				add(", CacheWritePerMTok: %s", num(p.CacheWritePerMTok))
			}
			if p.CacheReadPerMTok > 0 {
				add(", CacheReadPerMTok: %s", num(p.CacheReadPerMTok))
			}
			if p.ReasoningPerMTok > 0 {
				add(", ReasoningPerMTok: %s", num(p.ReasoningPerMTok))
			}
			add("},\n")
		}
		add("\n")
	}
	add("}\n\n")

	add("// defaultsAsOf is when the table above was generated. Exposed through\n")
	add("// AsOf when no remote snapshot is installed, so a caller can tell how\n")
	add("// old the rates behind a cost figure are.\n")
	add("var defaultsAsOf = %q\n", asOf.UTC().Format(time.RFC3339))

	return format.Source(b)
}

// num renders a rate as the shortest clean decimal.
//
// Upstream quotes per-token, so every rate arrives having been multiplied
// by a million: 2e-07 becomes 0.19999999999999998. Printed verbatim that
// is unreadable and makes every regeneration a noisy diff. Rates are
// published to at most a few decimal places, so rounding to ten recovers
// the number the vendor actually quoted without touching any real value.
func num(f float64) string {
	return strconv.FormatFloat(math.Round(f*1e10)/1e10, 'g', -1, 64)
}
