package pricing

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
)

// perMTok converts a per-token rate to the per-million-token rate this
// package works in.
//
// Both upstream feeds quote per-token, which puts a factor of a million
// between their numbers and ours. Getting this wrong is invisible in a
// diff and catastrophic in a bill, so it lives in one named function with
// its own test rather than being inlined at each call site.
func perMTok(perToken float64) float64 { return perToken * 1_000_000 }

// LiteLLMSource reads the community price table maintained alongside
// LiteLLM. It is the broadest machine-readable source available: several
// thousand models, updated close to daily, and — unusually — it carries
// cache-read and cache-write rates as well as context windows.
type LiteLLMSource struct {
	// URL overrides the default endpoint. Useful for pinning a commit or
	// pointing at an internal mirror.
	URL string
}

const liteLLMURL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

// Name identifies this source in errors and diagnostics.
func (LiteLLMSource) Name() string { return "litellm" }

// liteLLMEntry is the subset of LiteLLM's per-model object we use. The
// file carries embedding, image, and audio models too; mode lets us skip
// them.
type liteLLMEntry struct {
	InputCostPerToken  float64 `json:"input_cost_per_token"`
	OutputCostPerToken float64 `json:"output_cost_per_token"`
	CacheCreationCost  float64 `json:"cache_creation_input_token_cost"`
	CacheReadCost      float64 `json:"cache_read_input_token_cost"`
	Mode               string  `json:"mode"`
}

// Fetch downloads and parses the LiteLLM price table. Non-chat models and
// entries missing either rate are skipped rather than defaulted to zero.
func (s LiteLLMSource) Fetch(ctx context.Context, c *http.Client) (map[string]Price, error) {
	url := s.URL
	if url == "" {
		url = liteLLMURL
	}
	body, err := get(ctx, c, url)
	if err != nil {
		return nil, err
	}

	// Values are heterogeneous — the file has a "sample_spec" key whose
	// shape differs — so decode per entry and skip what does not fit
	// rather than failing the whole fetch.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("litellm: decode: %w", err)
	}

	out := make(map[string]Price, len(raw))
	for id, msg := range raw {
		var e liteLLMEntry
		if err := json.Unmarshal(msg, &e); err != nil {
			continue
		}
		if e.Mode != "" && e.Mode != "chat" && e.Mode != "responses" {
			continue
		}
		if e.InputCostPerToken <= 0 || e.OutputCostPerToken <= 0 {
			continue
		}
		out[id] = Price{
			InputPerMTok:      perMTok(e.InputCostPerToken),
			OutputPerMTok:     perMTok(e.OutputCostPerToken),
			CacheWritePerMTok: perMTok(e.CacheCreationCost),
			CacheReadPerMTok:  perMTok(e.CacheReadCost),
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("litellm: no chat models in %d entries", len(raw))
	}
	return out, nil
}

// OpenRouterSource reads OpenRouter's public model catalogue. Narrower
// than LiteLLM but first-party to a live gateway, so it tends to carry
// newly released models sooner.
type OpenRouterSource struct {
	// URL overrides the default endpoint.
	URL string
}

const openRouterURL = "https://openrouter.ai/api/v1/models"

// Name identifies this source in errors and diagnostics.
func (OpenRouterSource) Name() string { return "openrouter" }

// openRouterModels is OpenRouter's response envelope. Rates arrive as
// decimal *strings*, not numbers, so they need parsing rather than a
// plain float field.
type openRouterModels struct {
	Data []struct {
		ID      string `json:"id"`
		Pricing struct {
			Prompt          string `json:"prompt"`
			Completion      string `json:"completion"`
			InputCacheRead  string `json:"input_cache_read"`
			InputCacheWrite string `json:"input_cache_write"`
		} `json:"pricing"`
	} `json:"data"`
}

// Fetch downloads and parses OpenRouter's model catalogue. Rates arrive
// as decimal strings; anything unparseable is skipped rather than read as
// a free model.
func (s OpenRouterSource) Fetch(ctx context.Context, c *http.Client) (map[string]Price, error) {
	url := s.URL
	if url == "" {
		url = openRouterURL
	}
	body, err := get(ctx, c, url)
	if err != nil {
		return nil, err
	}

	var parsed openRouterModels
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("openrouter: decode: %w", err)
	}

	out := make(map[string]Price, len(parsed.Data))
	for _, m := range parsed.Data {
		in := parseRate(m.Pricing.Prompt)
		outRate := parseRate(m.Pricing.Completion)
		if in <= 0 || outRate <= 0 {
			continue
		}
		out[m.ID] = Price{
			InputPerMTok:      perMTok(in),
			OutputPerMTok:     perMTok(outRate),
			CacheReadPerMTok:  perMTok(parseRate(m.Pricing.InputCacheRead)),
			CacheWritePerMTok: perMTok(parseRate(m.Pricing.InputCacheWrite)),
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("openrouter: no priced models in %d entries", len(parsed.Data))
	}
	return out, nil
}

// parseRate reads OpenRouter's string-encoded per-token rate. Absent and
// unparseable both mean "no rate", which vet treats as a skip rather than
// a zero.
func parseRate(s string) float64 {
	if s == "" {
		return 0
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || v < 0 {
		return 0
	}
	return v
}

// maxBody caps how much of a price feed we will read. LiteLLM's table is
// a few megabytes; anything past this is a wrong URL or a hostile
// response, not a price list.
const maxBody = 32 << 20

func get(ctx context.Context, c *http.Client, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "llmgate-pricing")

	resp, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close() //nolint:errcheck // read-only response

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	return io.ReadAll(io.LimitReader(resp.Body, maxBody))
}
