// Package typesafe implements llmgate.Judge against TypeSafe's System One
// API, whose flagship model is Jev.
//
// It lives under judge/ rather than provider/ on purpose. Everything in
// provider/ adapts a chat-completion API onto llmgate.Client; nothing here
// implements Client, and a reader scanning provider/ for one should not
// find this.
//
// # What the model is for
//
// Jev returns typed judgments with calibrated probabilities. It does not
// generate text. Give it the decisions — route this, rank these, does this
// hold — and keep generation, arithmetic, counting, date comparison, and
// exact lookups in code, where they belong anyway.
//
// # The property that makes it worth using
//
// Questions in one request are answered in parallel against a single
// reading of the state, so a batch of forty questions is one round-trip
// rather than forty. Pack the batch; do not fan out.
package typesafe

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/pkoukk/tiktoken-go"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/pricing"
)

const (
	// DefaultBaseURL is TypeSafe's public API root.
	DefaultBaseURL = "https://api.typesafe.ai"

	// DefaultModel is the alias TypeSafe documents as the current stable
	// release.
	//
	// It is an alias, which means it moves. Once thresholds are tuned
	// against a specific version, pin that version's ID instead: the
	// answers behind an alias can change without any change here, and a
	// confidence threshold calibrated on one version is not evidence about
	// the next.
	DefaultModel = "jev-latest"

	// EnvAPIKey is the environment variable read when Config.APIKey is
	// empty.
	EnvAPIKey = "TYPESAFE_API_KEY"

	// evaluatePath is the System One evaluation endpoint.
	evaluatePath = "/v1/systemone"
)

// Context limits, as documented for jev-1.13 on 2026-09-17.
//
// The model ingests the state once and then processes every question
// against it, which is why there are two limits rather than one: a total
// budget, and a per-question budget that the largest single question has to
// fit inside alongside the state.
const (
	// MaxTotalTokens caps the state plus every question together.
	MaxTotalTokens = 64_000

	// MaxStateAndQuestionTokens caps the state plus the single longest
	// question.
	MaxStateAndQuestionTokens = 32_000
)

// Config configures a Judge.
type Config struct {
	// APIKey authenticates the request. Empty reads EnvAPIKey.
	APIKey string

	// BaseURL overrides the API root. Empty uses DefaultBaseURL.
	BaseURL string

	// Model is the default model for requests that do not name one.
	// Empty uses DefaultModel.
	Model string

	// HTTPClient overrides the HTTP client. Empty uses a client with a
	// 60s timeout.
	HTTPClient *http.Client

	// CountTokens overrides the token estimator used by the context guard.
	// Empty uses a cl100k_base estimate.
	//
	// Supply an exact counter if one becomes available: the default is an
	// approximation for a model whose tokenizer is not published, so the
	// guard is a cheap early warning, not an authority. See EstimateTokens.
	CountTokens func(string) (int, error)

	// SkipContextGuard disables the local size check, sending the request
	// and letting the server rule on it.
	//
	// Worth setting if the estimator ever rejects a request the API would
	// have accepted — a false rejection is the one failure mode of an
	// approximate guard, and it is better to lose the warning than to lose
	// the call.
	SkipContextGuard bool
}

// Judge is a System One client implementing llmgate.Judge.
type Judge struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
	countFn    func(string) (int, error)
	skipGuard  bool
}

// Compile-time check that the transport satisfies the seam.
var _ llmgate.Judge = (*Judge)(nil)

// New builds a Judge from cfg.
//
// It fails when no API key is available, rather than deferring to a 401 on
// the first call: a missing key is a deployment mistake, and surfacing it at
// construction puts the error where the fix is.
func New(cfg Config) (*Judge, error) {
	key := cfg.APIKey
	if key == "" {
		key = os.Getenv(EnvAPIKey)
	}
	if key == "" {
		return nil, fmt.Errorf("typesafe: no API key: set Config.APIKey or %s", EnvAPIKey)
	}

	base := cfg.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	base = strings.TrimRight(base, "/")

	model := cfg.Model
	if model == "" {
		model = DefaultModel
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 60 * time.Second}
	}

	countFn := cfg.CountTokens
	if countFn == nil {
		countFn = EstimateTokens
	}

	return &Judge{
		apiKey:     key,
		baseURL:    base,
		model:      model,
		httpClient: httpClient,
		countFn:    countFn,
		skipGuard:  cfg.SkipContextGuard,
	}, nil
}

// wireRequest is the JSON body of an evaluation call.
type wireRequest struct {
	State     any                         `json:"state"`
	Model     string                      `json:"model"`
	Questions map[string]llmgate.Question `json:"questions"`
}

// wireResponse is the JSON body of a successful evaluation.
//
// Answers stay raw until their `type` is known, because the three
// primitives do not share a shape.
type wireResponse struct {
	Model   string                     `json:"model"`
	Answers map[string]json.RawMessage `json:"answers"`
	Usage   struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// Judge evaluates req.State against every question in one HTTP request.
func (j *Judge) Judge(ctx context.Context, req llmgate.JudgeRequest) (*llmgate.Judgment, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	model := req.Model
	if model == "" {
		model = j.model
	}

	body, err := json.Marshal(wireRequest{
		State:     req.State,
		Model:     model,
		Questions: req.Questions,
	})
	if err != nil {
		return nil, fmt.Errorf("typesafe: marshal request: %w", err)
	}

	if !j.skipGuard {
		if err := j.checkContextBudget(req); err != nil {
			return nil, err
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, j.baseURL+evaluatePath, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("typesafe: build request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+j.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := j.httpClient.Do(httpReq)
	if err != nil {
		// A transport failure has no status code; classify it as transient
		// so the retry middleware treats a dropped connection the way it
		// treats a 503.
		return nil, &llmgate.LLMError{
			Class:    llmgate.ErrClassTransient,
			Provider: llmgate.ProviderTypeSafe,
			Message:  "request failed",
			Cause:    err,
		}
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &llmgate.LLMError{
			Class:    llmgate.ErrClassTransient,
			Provider: llmgate.ProviderTypeSafe,
			Message:  "read response body",
			Cause:    err,
		}
	}

	if resp.StatusCode != http.StatusOK {
		return nil, httpError(resp, raw)
	}

	var wire wireResponse
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, &llmgate.LLMError{
			Class:      llmgate.ErrClassGateway,
			StatusCode: resp.StatusCode,
			Provider:   llmgate.ProviderTypeSafe,
			Message:    "response was not valid JSON — a gateway or proxy likely answered instead of the API",
			Cause:      err,
		}
	}

	answers, err := decodeAnswers(wire.Answers)
	if err != nil {
		return nil, err
	}

	return &llmgate.Judgment{
		// The model the server reports, not the one that was asked for. An
		// alias resolves server-side, and the resolved ID is what a caller
		// needs in order to know which version produced a given threshold.
		Model:   wire.Model,
		Answers: answers,
		Usage:   usageFor(wire),
	}, nil
}

// usageFor normalizes the provider's counts onto llmgate.Usage and prices
// the call.
//
// Two zeros meet here and they mean opposite things, which is the whole
// reason this function is not a one-liner:
//
// Jev serves output tokens free, so an OutputTokens cost of zero is a real
// published rate. A caller must not read it as a gap in the price table.
//
// A response that reported no token counts at all is the other zero. Pricing
// that as $0 with Priced=true asserts the call was free, which it never is —
// the bug HAL-531 fixed. So Priced stays false whenever the counts did not
// come from the provider, however well-known the model's rate is.
func usageFor(wire wireResponse) llmgate.Usage {
	reported := wire.Usage.InputTokens > 0 || wire.Usage.OutputTokens > 0

	u := llmgate.Usage{
		InputTokens:    wire.Usage.InputTokens,
		OutputTokens:   wire.Usage.OutputTokens,
		TotalTokens:    wire.Usage.InputTokens + wire.Usage.OutputTokens,
		TokensReported: reported,
	}

	if !reported {
		return u
	}

	// Price against the model the server said answered, not the alias the
	// request may have named: an alias resolves server-side and can move.
	if cost, ok := pricing.ComputeWithOK(wire.Model, u.InputTokens, u.OutputTokens); ok {
		u.CostUSD = cost
		u.Priced = true
	}
	return u
}

// decodeAnswers turns the raw per-answer JSON into typed answers.
func decodeAnswers(raw map[string]json.RawMessage) (map[string]llmgate.Answer, error) {
	if len(raw) == 0 {
		return map[string]llmgate.Answer{}, nil
	}
	out := make(map[string]llmgate.Answer, len(raw))
	// Sorted so a malformed batch names the same answer on every run.
	ids := make([]string, 0, len(raw))
	for id := range raw {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	for _, id := range ids {
		a, err := decodeAnswer(raw[id])
		if err != nil {
			return nil, fmt.Errorf("typesafe: answer %q: %w", id, err)
		}
		out[id] = a
	}
	return out, nil
}

// decodeAnswer dispatches on the answer's `type` discriminator.
func decodeAnswer(raw json.RawMessage) (llmgate.Answer, error) {
	var probe struct {
		Type llmgate.QuestionKind `json:"type"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return nil, fmt.Errorf("read answer type: %w", err)
	}

	switch probe.Type {
	case llmgate.KindNoul:
		var a struct {
			Noul float64 `json:"noul"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return nil, fmt.Errorf("decode noul: %w", err)
		}
		return llmgate.NoulAnswer{Noul: a.Noul}, nil

	case llmgate.KindChoice:
		var a struct {
			Choice        string             `json:"choice"`
			Probabilities map[string]float64 `json:"probabilities"`
			Confidence    float64            `json:"confidence"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return nil, fmt.Errorf("decode choice: %w", err)
		}
		return llmgate.ChoiceAnswer{
			Choice:        a.Choice,
			Probabilities: a.Probabilities,
			Confidence:    a.Confidence,
		}, nil

	case llmgate.KindScore:
		var a struct {
			Score         float64            `json:"score"`
			Legend        map[string]string  `json:"legend"`
			Probabilities map[string]float64 `json:"probabilities"`
			Confidence    float64            `json:"confidence"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return nil, fmt.Errorf("decode score: %w", err)
		}
		legend, err := intKeyedStrings(a.Legend)
		if err != nil {
			return nil, fmt.Errorf("decode score legend: %w", err)
		}
		probs, err := intKeyedFloats(a.Probabilities)
		if err != nil {
			return nil, fmt.Errorf("decode score probabilities: %w", err)
		}
		return llmgate.ScoreAnswer{
			Score:         a.Score,
			Legend:        legend,
			Probabilities: probs,
			Confidence:    a.Confidence,
		}, nil

	case "":
		return nil, fmt.Errorf("answer has no type field")
	default:
		// A primitive this package does not know about. Fail loudly rather
		// than dropping it: a silently missing answer would surface much
		// later as an ErrAnswerMissing from an accessor, a long way from
		// the cause.
		return nil, fmt.Errorf("unknown answer type %q", probe.Type)
	}
}

// intKeyedStrings converts a wire legend, whose keys are stringified level
// indices, into an int-keyed map.
func intKeyedStrings(in map[string]string) (map[int]string, error) {
	if in == nil {
		return nil, nil
	}
	out := make(map[int]string, len(in))
	for k, v := range in {
		i, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("level key %q is not an integer", k)
		}
		out[i] = v
	}
	return out, nil
}

// intKeyedFloats converts a wire distribution over score levels into an
// int-keyed map.
func intKeyedFloats(in map[string]float64) (map[int]float64, error) {
	if in == nil {
		return nil, nil
	}
	out := make(map[int]float64, len(in))
	for k, v := range in {
		i, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("level key %q is not an integer", k)
		}
		out[i] = v
	}
	return out, nil
}

// httpError maps a non-200 response onto llmgate's error taxonomy.
//
// The classes are set explicitly rather than derived from the status code,
// because 422 and 529 both fall through llmgate's generic mapping. 422 in
// particular would land on Unknown, which the retry middleware retries by
// default — and a validation failure is deterministic, so retrying it burns
// the budget four times to be told the same thing.
func httpError(resp *http.Response, body []byte) error {
	msg := errorMessage(body)

	e := &llmgate.LLMError{
		StatusCode: resp.StatusCode,
		Provider:   llmgate.ProviderTypeSafe,
		Message:    msg,
	}

	switch {
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		e.Class = llmgate.ErrClassAuth
	case resp.StatusCode == http.StatusUnprocessableEntity:
		e.Class = llmgate.ErrClassBadRequest
	case resp.StatusCode == http.StatusTooManyRequests:
		e.Class = llmgate.ErrClassRateLimited
		e.RetryAfterDur = retryAfter(resp.Header)
	case resp.StatusCode >= 500:
		// 529 Overloaded lands here alongside the ordinary 5xx family.
		e.Class = llmgate.ErrClassTransient
	default:
		e.Class = llmgate.ErrClassUnknown
	}
	return e
}

// errorMessage pulls a human-readable message out of an error body,
// falling back to the raw text when it is not the shape we expect.
func errorMessage(body []byte) string {
	trimmed := strings.TrimSpace(string(body))
	if trimmed == "" {
		return "no error body"
	}

	var payload struct {
		Message string `json:"message"`
		Detail  any    `json:"detail"`
		Error   string `json:"error"`
	}
	if err := json.Unmarshal(body, &payload); err == nil {
		switch {
		case payload.Message != "":
			return payload.Message
		case payload.Error != "":
			return payload.Error
		case payload.Detail != nil:
			// A 422 names the offending field in detail; keeping it is the
			// difference between a fixable error and a shrug.
			if d, err := json.Marshal(payload.Detail); err == nil {
				return string(d)
			}
		}
	}
	return truncate(trimmed, 512)
}

// retryAfter reads the Retry-After header, which the provider sends on a
// 429 to say exactly when capacity frees up. Honouring it beats guessing:
// backoff either sleeps too little and earns another 429, or too much and
// wastes wall-clock.
func retryAfter(h http.Header) time.Duration {
	v := strings.TrimSpace(h.Get("Retry-After"))
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil {
		if secs <= 0 {
			return 0
		}
		return time.Duration(secs) * time.Second
	}
	// The header also permits an HTTP-date.
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}

// checkContextBudget rejects a request that cannot fit, before it is sent.
//
// Two limits, because the model reads the state once and then runs every
// question against it: a total budget, and a per-question budget the
// largest single question must fit inside alongside the state.
//
// The estimate is approximate — Jev's tokenizer is not published, so this
// uses cl100k_base. It is deliberately a guard and not an authority: the
// server rules on the real thing, and Config.SkipContextGuard exists for
// the case where an approximation rejects something the API would accept.
func (j *Judge) checkContextBudget(req llmgate.JudgeRequest) error {
	stateJSON, err := json.Marshal(req.State)
	if err != nil {
		return fmt.Errorf("typesafe: marshal state for budget check: %w", err)
	}
	stateTokens, err := j.countFn(string(stateJSON))
	if err != nil {
		// A broken estimator must not block a request that may well be
		// fine. Skip the guard and let the server decide.
		return nil
	}

	total := stateTokens
	longest := 0
	longestID := ""

	for _, id := range sortedQuestionIDs(req.Questions) {
		qJSON, err := json.Marshal(req.Questions[id])
		if err != nil {
			return fmt.Errorf("typesafe: marshal question %q for budget check: %w", id, err)
		}
		n, err := j.countFn(string(qJSON))
		if err != nil {
			return nil
		}
		total += n
		if n > longest {
			longest, longestID = n, id
		}
	}

	if total > MaxTotalTokens {
		return &llmgate.LLMError{
			Class:    llmgate.ErrClassContextLength,
			Provider: llmgate.ProviderTypeSafe,
			Message: fmt.Sprintf(
				"state plus %d questions is about %d tokens, over the %d limit — send fewer questions per call, or trim the state",
				len(req.Questions), total, MaxTotalTokens),
		}
	}

	if stateTokens+longest > MaxStateAndQuestionTokens {
		return &llmgate.LLMError{
			Class:    llmgate.ErrClassContextLength,
			Provider: llmgate.ProviderTypeSafe,
			Message: fmt.Sprintf(
				"state plus the longest question (%q) is about %d tokens, over the %d per-question limit",
				longestID, stateTokens+longest, MaxStateAndQuestionTokens),
		}
	}
	return nil
}

// sortedQuestionIDs keeps budget accounting and its error messages
// deterministic across runs.
func sortedQuestionIDs(q map[string]llmgate.Question) []string {
	ids := make([]string, 0, len(q))
	for id := range q {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// EstimateTokens approximates the token count of text using cl100k_base.
//
// It is an estimate. TypeSafe does not publish Jev's tokenizer, so this
// borrows the closest widely-available one; expect it to be in the right
// neighbourhood rather than exact. It exists to catch a request that is
// obviously too large before it costs a round-trip, not to predict the
// bill.
func EstimateTokens(text string) (int, error) {
	enc, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		return 0, fmt.Errorf("typesafe: load tokenizer: %w", err)
	}
	return len(enc.Encode(text, nil, nil)), nil
}

// truncate shortens s to at most n characters, marking that it was cut.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
