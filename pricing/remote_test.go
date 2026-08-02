package pricing_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/hallelx2/llmgate/pricing"
)

// serveJSON stands in for an upstream feed. Every test in this file runs
// against it — the package must never reach the real network during
// `go test`, or CI becomes dependent on GitHub and OpenRouter being up.
func serveJSON(t *testing.T, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// liteLLMFixture mirrors the real file's shape, including the non-chat
// entries and the odd sample_spec key that must be skipped.
const liteLLMFixture = `{
  "sample_spec": {"note": "this is not a model"},
  "claude-sonnet-4-5": {
    "input_cost_per_token": 0.000003,
    "output_cost_per_token": 0.000015,
    "cache_creation_input_token_cost": 0.00000375,
    "cache_read_input_token_cost": 0.0000003,
    "mode": "chat"
  },
  "zhipuai/glm-4.6": {
    "input_cost_per_token": 0.0000006,
    "output_cost_per_token": 0.0000022,
    "mode": "chat"
  },
  "text-embedding-3-small": {
    "input_cost_per_token": 0.00000002,
    "output_cost_per_token": 0.0,
    "mode": "embedding"
  }
}`

// openRouterFixture uses string-encoded per-token rates, as the real API
// does.
const openRouterFixture = `{"data":[
  {"id":"z-ai/glm-4.6","pricing":{"prompt":"0.0000006","completion":"0.0000022","input_cache_read":"0.00000011"}},
  {"id":"openai/gpt-4o","pricing":{"prompt":"0.0000025","completion":"0.00001"}},
  {"id":"broken/no-price","pricing":{"prompt":"","completion":""}}
]}`

func TestLiteLLMSourceParses(t *testing.T) {
	srv := serveJSON(t, liteLLMFixture)
	got, err := pricing.LiteLLMSource{URL: srv.URL}.Fetch(context.Background(), srv.Client())
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}

	// Per-token in, per-Mtok out. An error here is a factor of a million.
	sonnet, ok := got["claude-sonnet-4-5"]
	if !ok {
		t.Fatal("claude-sonnet-4-5 missing from parsed feed")
	}
	if sonnet.InputPerMTok != 3.0 || sonnet.OutputPerMTok != 15.0 {
		t.Errorf("sonnet rates = %+v, want 3.00/15.00 per Mtok", sonnet)
	}
	if sonnet.CacheWritePerMTok != 3.75 || sonnet.CacheReadPerMTok != 0.30 {
		t.Errorf("sonnet cache rates = %+v, want 3.75/0.30", sonnet)
	}

	if _, ok := got["sample_spec"]; ok {
		t.Error("sample_spec is not a model and must be skipped")
	}
	if _, ok := got["text-embedding-3-small"]; ok {
		t.Error("embedding models must be skipped — they have no output rate")
	}
}

func TestOpenRouterSourceParses(t *testing.T) {
	srv := serveJSON(t, openRouterFixture)
	got, err := pricing.OpenRouterSource{URL: srv.URL}.Fetch(context.Background(), srv.Client())
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}

	glm, ok := got["z-ai/glm-4.6"]
	if !ok {
		t.Fatal("z-ai/glm-4.6 missing from parsed feed")
	}
	if glm.InputPerMTok != 0.6 || glm.OutputPerMTok != 2.2 {
		t.Errorf("glm rates = %+v, want 0.60/2.20 per Mtok", glm)
	}
	if glm.CacheReadPerMTok != 0.11 {
		t.Errorf("glm cache-read = %v, want 0.11", glm.CacheReadPerMTok)
	}

	if _, ok := got["broken/no-price"]; ok {
		t.Error("an entry with empty rate strings must be skipped, not parsed as free")
	}
}

// TestUseRemoteLayering is the core contract: Register beats remote,
// remote beats the embedded table, and an unknown model still falls
// through to unpriced.
func TestUseRemoteLayering(t *testing.T) {
	// A feed that disagrees with the embedded table by a believable
	// amount, plus a model the embedded table has never heard of.
	srv := serveJSON(t, `{
      "claude-haiku-4-5": {"input_cost_per_token": 0.000002, "output_cost_per_token": 0.00001, "mode": "chat"},
      "brand-new-model-9": {"input_cost_per_token": 0.000007, "output_cost_per_token": 0.000021, "mode": "chat"}
    }`)

	stop, err := pricing.UseRemote(context.Background(), pricing.RemoteConfig{
		Sources:         []pricing.Source{pricing.LiteLLMSource{URL: srv.URL}},
		RefreshInterval: -1, // fetch once, no goroutine
		HTTPClient:      srv.Client(),
	})
	if err != nil {
		t.Fatalf("UseRemote: %v", err)
	}
	t.Cleanup(stop)

	// Remote beats the embedded table.
	if p, _ := pricing.Lookup("claude-haiku-4-5"); p.InputPerMTok != 2.0 {
		t.Errorf("claude-haiku-4-5 input = %v, want the remote 2.00 (embedded is 1.00)", p.InputPerMTok)
	}

	// Remote supplies models the embedded table lacks.
	if p, ok := pricing.Lookup("brand-new-model-9"); !ok || p.OutputPerMTok != 21.0 {
		t.Errorf("brand-new-model-9 = %+v ok=%v, want output 21.00 from remote", p, ok)
	}

	// Register still wins over remote. Undo it afterwards — a leaked
	// override would silently change what every later test prices.
	pinned := pricing.Price{InputPerMTok: 42, OutputPerMTok: 84}
	pricing.Register("claude-haiku-4-5", pinned)
	t.Cleanup(func() { pricing.Unregister("claude-haiku-4-5") })
	if p, _ := pricing.Lookup("claude-haiku-4-5"); p != pinned {
		t.Errorf("after Register, lookup = %+v, want the pinned %+v", p, pinned)
	}

	// Models in neither remote nor embedded stay unpriced.
	if _, ok := pricing.Lookup("still-nonexistent-zzz"); ok {
		t.Error("an unknown model must stay unpriced even with a remote layer")
	}

	if when, src := pricing.AsOf(); when.IsZero() || src == "" {
		t.Errorf("AsOf = (%v, %q), want a populated vintage", when, src)
	}
}

// TestRemoteRejectsImplausibleRates guards the units error that would
// otherwise be invisible: a feed quoting per-Mtok where we expect
// per-token misreports every cost by a factor of a million.
func TestRemoteRejectsImplausibleRates(t *testing.T) {
	// gpt-4o at "2.5 per token" — i.e. someone forgot the conversion.
	srv := serveJSON(t, `{"gpt-4o": {"input_cost_per_token": 2.5, "output_cost_per_token": 10, "mode": "chat"}}`)

	var errs []string
	stop, err := pricing.UseRemote(context.Background(), pricing.RemoteConfig{
		Sources:         []pricing.Source{pricing.LiteLLMSource{URL: srv.URL}},
		RefreshInterval: -1,
		HTTPClient:      srv.Client(),
		OnError:         func(src string, e error) { errs = append(errs, src+": "+e.Error()) },
	})
	if err != nil {
		t.Fatalf("UseRemote: %v", err)
	}
	t.Cleanup(stop)

	if len(errs) == 0 {
		t.Fatal("an implausible rate must be reported through OnError")
	}
	// The embedded rate must survive the rejected snapshot.
	if p, _ := pricing.Lookup("gpt-4o"); p.InputPerMTok != 2.50 {
		t.Errorf("gpt-4o input = %v, want the embedded 2.50 — a corrupt snapshot must not be adopted", p.InputPerMTok)
	}
}

// TestRemoteFallsThroughSources: a dead first source must not stop a
// healthy second one.
func TestRemoteFallsThroughSources(t *testing.T) {
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(dead.Close)
	good := serveJSON(t, `{"data":[{"id":"fallthrough-model","pricing":{"prompt":"0.000001","completion":"0.000002"}}]}`)

	var errs int
	stop, err := pricing.UseRemote(context.Background(), pricing.RemoteConfig{
		Sources: []pricing.Source{
			pricing.LiteLLMSource{URL: dead.URL},
			pricing.OpenRouterSource{URL: good.URL},
		},
		RefreshInterval: -1,
		HTTPClient:      good.Client(),
		OnError:         func(string, error) { errs++ },
	})
	if err != nil {
		t.Fatalf("UseRemote: %v", err)
	}
	t.Cleanup(stop)

	if errs != 1 {
		t.Errorf("OnError fired %d times, want 1 for the dead source", errs)
	}
	if _, ok := pricing.Lookup("fallthrough-model"); !ok {
		t.Error("the second source's models must load when the first fails")
	}
}

// TestRemoteSurvivesTotalOutage: with every source down, lookups keep
// working off the embedded table. A price refresh must never be able to
// break a completion.
func TestRemoteSurvivesTotalOutage(t *testing.T) {
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	t.Cleanup(dead.Close)

	stop, err := pricing.UseRemote(context.Background(), pricing.RemoteConfig{
		Sources:         []pricing.Source{pricing.LiteLLMSource{URL: dead.URL}},
		RefreshInterval: -1,
		HTTPClient:      dead.Client(),
	})
	if err != nil {
		t.Fatalf("UseRemote must not return an error when sources are down: %v", err)
	}
	t.Cleanup(stop)

	if p, ok := pricing.Lookup("glm-4.6"); !ok || p.InputPerMTok <= 0 {
		t.Error("embedded prices must still resolve when every remote source is down")
	}
}

// TestRemoteDiskCache: a snapshot persists and is reloaded, so a restart
// starts warm rather than falling back to the embedded table.
func TestRemoteDiskCache(t *testing.T) {
	dir := t.TempDir()
	srv := serveJSON(t, `{"cached-only-model": {"input_cost_per_token": 0.000009, "output_cost_per_token": 0.000018, "mode": "chat"}}`)

	stop, err := pricing.UseRemote(context.Background(), pricing.RemoteConfig{
		Sources:         []pricing.Source{pricing.LiteLLMSource{URL: srv.URL}},
		RefreshInterval: -1,
		CacheDir:        dir,
		HTTPClient:      srv.Client(),
	})
	if err != nil {
		t.Fatalf("UseRemote: %v", err)
	}
	stop()

	// Second start, with every source dead: the cache must supply it.
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(dead.Close)

	stop2, err := pricing.UseRemote(context.Background(), pricing.RemoteConfig{
		Sources:         []pricing.Source{pricing.LiteLLMSource{URL: dead.URL}},
		RefreshInterval: -1,
		CacheDir:        dir,
		HTTPClient:      dead.Client(),
	})
	if err != nil {
		t.Fatalf("UseRemote (cached start): %v", err)
	}
	t.Cleanup(stop2)

	p, ok := pricing.Lookup("cached-only-model")
	if !ok {
		t.Fatal("a persisted snapshot must be reloaded on start")
	}
	if p.InputPerMTok != 9.0 {
		t.Errorf("cached rate = %v, want 9.00", p.InputPerMTok)
	}
}

// TestLookupDoesNotBlockOnNetwork: lookups read an atomically swapped
// snapshot, never the network. A hung feed must not stall a completion.
func TestLookupDoesNotBlockOnNetwork(t *testing.T) {
	hang := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	t.Cleanup(hang.Close)

	done := make(chan struct{})
	go func() {
		defer close(done)
		stop, _ := pricing.UseRemote(context.Background(), pricing.RemoteConfig{
			Sources:         []pricing.Source{pricing.LiteLLMSource{URL: hang.URL}},
			RefreshInterval: -1,
			Timeout:         300 * time.Millisecond,
			HTTPClient:      hang.Client(),
		})
		stop()
	}()

	// While that fetch hangs, ordinary lookups must return immediately.
	for range 100 {
		if _, ok := pricing.Lookup("glm-4.6"); !ok {
			t.Fatal("lookup failed during a hanging refresh")
		}
	}

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("UseRemote did not give up on a hanging source within its timeout")
	}
}
