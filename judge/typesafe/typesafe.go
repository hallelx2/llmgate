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
	"net"
	"net/http"
	"net/http/httptrace"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
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

	// OnRequest, when set, is called once per Judge call — success or
	// failure — with where the time went. It runs on the calling
	// goroutine after the call completes, so it must be cheap and must
	// not block.
	//
	// It exists because a round-trip number alone cannot tell local CPU
	// from provider latency: the 2026-09-25 query evaluation measured
	// 28.6 s against a predicted ~9 s, and the difference was the guard
	// rebuilding its tokenizer on every count (HAL-1708), inside the
	// concurrency limiter where it looked like the provider being slow.
	OnRequest func(RequestTrace)
}

// RequestTrace is where one Judge call's time went.
type RequestTrace struct {
	Model      string
	Questions  int
	StateBytes int // the marshalled state
	BodyBytes  int // the whole request body

	// Prepare covers validation, marshalling and the context guard —
	// everything local before the request goes on the wire.
	Prepare time.Duration
	// GuardCounted is true when the guard ran the tokenizer, false when
	// the byte bound proved the request fits (or the guard was skipped).
	GuardCounted bool

	// Connect is DNS + TCP + TLS for a new connection; zero when an idle
	// connection (or an HTTP/2 stream on one) was reused.
	Connect    time.Duration
	ConnReused bool
	// FirstByte runs from the request going out to the first response
	// byte: the provider's own time plus one network round trip.
	FirstByte time.Duration
	// Total is the whole call, Prepare included.
	Total time.Duration

	StatusCode  int
	InputTokens int
	Err         error
}

// Judge is a System One client implementing llmgate.Judge.
type Judge struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
	countFn    func(string) (int, error)
	// byteBound is true when countFn is the built-in estimator, whose
	// tokens each cover at least one byte — so a byte count is a proven
	// upper bound on the token count and the guard can skip counting
	// whenever bytes alone fit. A caller-supplied counter makes no such
	// promise, so it is always consulted.
	byteBound bool
	skipGuard bool
	onRequest func(RequestTrace)
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
	if err := checkBaseURL(base); err != nil {
		return nil, err
	}

	model := cfg.Model
	if model == "" {
		model = DefaultModel
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 60 * time.Second}
	}

	countFn := cfg.CountTokens
	byteBound := false
	if countFn == nil {
		countFn = EstimateTokens
		byteBound = true
	}

	return &Judge{
		apiKey:     key,
		baseURL:    base,
		model:      model,
		httpClient: httpClient,
		countFn:    countFn,
		byteBound:  byteBound,
		skipGuard:  cfg.SkipContextGuard,
		onRequest:  cfg.OnRequest,
	}, nil
}

// checkBaseURL refuses a base URL that would put the API key on the wire in
// cleartext.
//
// Every request carries `Authorization: Bearer <key>`, so a plain-http base
// leaks a live credential to anything on the path. That is worth failing at
// construction rather than trusting a caller to notice.
//
// Loopback is exempt: a test server and a local proxy are both http, and
// neither leaves the machine.
func checkBaseURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("typesafe: base URL %q is not a URL: %w", raw, err)
	}
	switch u.Scheme {
	case "https":
		return nil
	case "http":
		if isLoopback(u.Hostname()) {
			return nil
		}
		return fmt.Errorf(
			"typesafe: base URL %q is plain http, which would send the API key in cleartext; use https (http is allowed for loopback only)",
			raw)
	default:
		return fmt.Errorf("typesafe: base URL %q must be http or https, got scheme %q", raw, u.Scheme)
	}
}

// isLoopback reports whether host addresses this machine.
func isLoopback(host string) bool {
	if host == "localhost" {
		return true
	}
	if ip := net.ParseIP(host); ip != nil {
		return ip.IsLoopback()
	}
	return false
}

// maxResponseBytes caps how much of a response body is read.
//
// A judgment response is small — a few hundred bytes per answer. Reading
// without a bound means a broken or hostile endpoint can stream until the
// process dies, turning someone else's fault into an OOM here. 8 MiB is
// orders of magnitude above any real response and still bounded.
const maxResponseBytes = 8 << 20

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

	// The counts are pointers so a genuine zero can be told apart from a
	// field the provider omitted. Inferring presence from "is it positive"
	// would mark an explicit {"input_tokens":0,"output_tokens":0} as
	// unreported, which is the same conflation HAL-531 was about, just
	// pointing the other way.
	Usage struct {
		InputTokens  *int `json:"input_tokens"`
		OutputTokens *int `json:"output_tokens"`
	} `json:"usage"`
}

// Judge evaluates req.State against every question in one HTTP request.
func (j *Judge) Judge(ctx context.Context, req llmgate.JudgeRequest) (res *llmgate.Judgment, err error) {
	start := time.Now()
	tr := RequestTrace{Questions: len(req.Questions)}
	if j.onRequest != nil {
		defer func() {
			tr.Total = time.Since(start)
			tr.Err = err
			if res != nil {
				tr.InputTokens = res.Usage.InputTokens
			}
			j.onRequest(tr)
		}()
	}

	if err := req.Validate(); err != nil {
		return nil, err
	}

	model := req.Model
	if model == "" {
		model = j.model
	}
	tr.Model = model

	// The state is marshalled once: the body embeds it verbatim and the
	// guard measures it, rather than each serialising it separately.
	stateJSON, err := json.Marshal(req.State)
	if err != nil {
		return nil, fmt.Errorf("typesafe: marshal request: %w", err)
	}
	tr.StateBytes = len(stateJSON)

	body, err := json.Marshal(wireRequest{
		State:     json.RawMessage(stateJSON),
		Model:     model,
		Questions: req.Questions,
	})
	if err != nil {
		return nil, fmt.Errorf("typesafe: marshal request: %w", err)
	}
	tr.BodyBytes = len(body)

	if !j.skipGuard {
		counted, err := j.checkContextBudget(stateJSON, req.Questions)
		tr.GuardCounted = counted
		if err != nil {
			return nil, err
		}
	}
	tr.Prepare = time.Since(start)

	// The transport calls these hooks on its own goroutines, and can still
	// be running one when Do returns with an error; the timings are held
	// under a lock and copied into the trace when the call ends.
	var (
		wireMu    sync.Mutex
		connStart time.Time
		sent      time.Time
		connect   time.Duration
		reused    bool
		firstByte time.Duration
	)
	if j.onRequest != nil {
		defer func() {
			wireMu.Lock()
			tr.Connect, tr.ConnReused, tr.FirstByte = connect, reused, firstByte
			wireMu.Unlock()
		}()
	}
	trace := &httptrace.ClientTrace{
		GetConn: func(string) {
			wireMu.Lock()
			connStart = time.Now()
			wireMu.Unlock()
		},
		GotConn: func(info httptrace.GotConnInfo) {
			wireMu.Lock()
			reused = info.Reused
			if !info.Reused && !connStart.IsZero() {
				connect = time.Since(connStart)
			}
			wireMu.Unlock()
		},
		WroteRequest: func(httptrace.WroteRequestInfo) {
			wireMu.Lock()
			sent = time.Now()
			wireMu.Unlock()
		},
		GotFirstResponseByte: func() {
			wireMu.Lock()
			if !sent.IsZero() {
				firstByte = time.Since(sent)
			}
			wireMu.Unlock()
		},
	}
	httpReq, err := http.NewRequestWithContext(httptrace.WithClientTrace(ctx, trace), http.MethodPost, j.baseURL+evaluatePath, bytes.NewReader(body))
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
	defer func() { _ = resp.Body.Close() }()
	tr.StatusCode = resp.StatusCode

	// Bounded: one extra byte past the cap tells us it overflowed rather
	// than merely reached the limit exactly.
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes+1))
	if err != nil {
		return nil, &llmgate.LLMError{
			Class:    llmgate.ErrClassTransient,
			Provider: llmgate.ProviderTypeSafe,
			Message:  "read response body",
			Cause:    err,
		}
	}
	if len(raw) > maxResponseBytes {
		return nil, &llmgate.LLMError{
			Class:      llmgate.ErrClassGateway,
			StatusCode: resp.StatusCode,
			Provider:   llmgate.ProviderTypeSafe,
			Message: fmt.Sprintf("response body exceeds %d bytes; a judgment response is never this large, so something else is answering",
				maxResponseBytes),
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
	if err := checkComplete(req.Questions, answers); err != nil {
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
	// Presence, not positivity: a provider that explicitly reports zero
	// tokens has still reported.
	reported := wire.Usage.InputTokens != nil || wire.Usage.OutputTokens != nil

	in := derefOr(wire.Usage.InputTokens, 0)
	out := derefOr(wire.Usage.OutputTokens, 0)

	u := llmgate.Usage{
		InputTokens:    in,
		OutputTokens:   out,
		TotalTokens:    in + out,
		TokensReported: reported,
	}

	if !reported {
		return u
	}

	// Price against the model the server said answered, not the alias the
	// request may have named: an alias resolves server-side and can move.
	// Jev has no prompt cache and no separate reasoning tier, so the
	// breakdown is just input and output — but this goes through
	// ComputeTokens rather than the deprecated two-int form, which bills a
	// cached prompt as if it were uncached.
	if cost, ok := pricing.ComputeTokens(wire.Model, pricing.Tokens{
		Input:  u.InputTokens,
		Output: u.OutputTokens,
	}); ok {
		u.CostUSD = cost
		u.Priced = true
	}
	return u
}

// checkComplete verifies every question came back answered.
//
// A partial 200 is otherwise silent: the judgment looks fine, and the gap
// only surfaces later as an ErrAnswerMissing from an accessor, a long way
// from the response that caused it. Failing at the transport boundary puts
// the error where the evidence is — and an unanswered question is not a
// degraded result a caller should quietly proceed on, since the code path
// that consumes it will branch on a zero.
func checkComplete(asked map[string]llmgate.Question, got map[string]llmgate.Answer) error {
	var problems []string

	for id, q := range asked {
		a, ok := got[id]
		if !ok {
			problems = append(problems, fmt.Sprintf("%q unanswered", id))
			continue
		}
		// A Noul answered with a Choice is not a usable answer under a
		// different name; it means the response does not correspond to the
		// request. Caught here, the caller sees why — left alone, it
		// surfaces as ErrAnswerType from an accessor much later.
		if a.Kind() != q.Kind() {
			problems = append(problems,
				fmt.Sprintf("%q was asked as %s but answered as %s", id, q.Kind(), a.Kind()))
		}
	}

	for id := range got {
		if _, ok := asked[id]; !ok {
			problems = append(problems, fmt.Sprintf("%q was not asked", id))
		}
	}

	if len(problems) == 0 {
		return nil
	}
	sort.Strings(problems)
	return &llmgate.LLMError{
		Class:    llmgate.ErrClassGateway,
		Provider: llmgate.ProviderTypeSafe,
		Message: fmt.Sprintf("response does not match the request (%d questions asked): %s",
			len(asked), strings.Join(problems, "; ")),
	}
}

// derefOr returns *p, or fallback when p is nil.
func derefOr[T any](p *T, fallback T) T {
	if p == nil {
		return fallback
	}
	return *p
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

	// Required numeric fields are decoded through pointers so an omitted
	// one is distinguishable from a real zero. This matters more here than
	// almost anywhere else in the package: a missing "noul" silently
	// becoming 0.0 does not look like an error downstream, it looks like a
	// confident "definitely no", and it will drive a threshold accordingly.
	switch probe.Type {
	case llmgate.KindNoul:
		var a struct {
			Noul *float64 `json:"noul"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return nil, fmt.Errorf("decode noul: %w", err)
		}
		if a.Noul == nil {
			return nil, fmt.Errorf("noul answer has no %q field", "noul")
		}
		if err := inUnitRange("noul", *a.Noul); err != nil {
			return nil, err
		}
		return llmgate.NoulAnswer{Noul: *a.Noul}, nil

	case llmgate.KindChoice:
		var a struct {
			Choice        *string            `json:"choice"`
			Probabilities map[string]float64 `json:"probabilities"`
			Confidence    *float64           `json:"confidence"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return nil, fmt.Errorf("decode choice: %w", err)
		}
		if a.Choice == nil || *a.Choice == "" {
			return nil, fmt.Errorf("choice answer has no selected option")
		}
		if len(a.Probabilities) == 0 {
			return nil, fmt.Errorf("choice answer has no probability distribution")
		}
		if a.Confidence == nil {
			return nil, fmt.Errorf("choice answer has no confidence")
		}
		if err := inUnitRange("confidence", *a.Confidence); err != nil {
			return nil, err
		}
		return llmgate.ChoiceAnswer{
			Choice:        *a.Choice,
			Probabilities: a.Probabilities,
			Confidence:    *a.Confidence,
		}, nil

	case llmgate.KindScore:
		var a struct {
			Score         *float64           `json:"score"`
			Legend        map[string]string  `json:"legend"`
			Probabilities map[string]float64 `json:"probabilities"`
			Confidence    *float64           `json:"confidence"`
		}
		if err := json.Unmarshal(raw, &a); err != nil {
			return nil, fmt.Errorf("decode score: %w", err)
		}
		if a.Score == nil {
			return nil, fmt.Errorf("score answer has no %q field", "score")
		}
		if len(a.Probabilities) == 0 {
			return nil, fmt.Errorf("score answer has no probability distribution")
		}
		if a.Confidence == nil {
			return nil, fmt.Errorf("score answer has no confidence")
		}
		if err := inUnitRange("confidence", *a.Confidence); err != nil {
			return nil, err
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
			Score:         *a.Score,
			Legend:        legend,
			Probabilities: probs,
			Confidence:    *a.Confidence,
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

// inUnitRange rejects a probability or confidence outside [0, 1].
//
// Out-of-range here means the body is not what it claims to be — a
// mangling proxy, or a shape change — and passing it through would hand a
// caller a "probability" of 7 to threshold against.
func inUnitRange(field string, v float64) error {
	if v < 0 || v > 1 {
		return fmt.Errorf("%s is %v, outside [0,1]", field, v)
	}
	return nil
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
// counted reports whether the tokenizer ran.
//
// Two limits, because the model reads the state once and then runs every
// question against it: a total budget, and a per-question budget the
// largest single question must fit inside alongside the state.
//
// The estimate is approximate — Jev's tokenizer is not published, so this
// uses cl100k_base. It is deliberately a guard and not an authority: the
// server rules on the real thing, and Config.SkipContextGuard exists for
// the case where an approximation rejects something the API would accept.
//
// It is also on every request's critical path, so it does the least work
// that can decide. With the built-in estimator every token covers at least
// one byte, so byte lengths are an upper bound on token counts: when the
// bytes fit, the tokens fit, and nothing is tokenised. Past that, the state
// is counted exactly once and each question is still bounded by its bytes;
// questions are tokenised only if the bound is not enough to decide.
func (j *Judge) checkContextBudget(stateJSON []byte, questions map[string]llmgate.Question) (counted bool, err error) {
	ids := sortedQuestionIDs(questions)
	qJSON := make([][]byte, len(ids))
	qBytes, longestBytes := 0, 0
	for i, id := range ids {
		b, err := json.Marshal(questions[id])
		if err != nil {
			return false, fmt.Errorf("typesafe: marshal question %q for budget check: %w", id, err)
		}
		qJSON[i] = b
		qBytes += len(b)
		longestBytes = max(longestBytes, len(b))
	}

	if j.byteBound {
		if len(stateJSON)+qBytes <= MaxTotalTokens && len(stateJSON)+longestBytes <= MaxStateAndQuestionTokens {
			return false, nil
		}
	}

	stateTokens, err := j.countFn(string(stateJSON))
	if err != nil {
		// A broken estimator must not block a request that may well be
		// fine. Skip the guard and let the server decide.
		return true, nil
	}
	if j.byteBound {
		if stateTokens+qBytes <= MaxTotalTokens && stateTokens+longestBytes <= MaxStateAndQuestionTokens {
			return true, nil
		}
	}

	total := stateTokens
	longest := 0
	longestID := ""
	for i, id := range ids {
		n, err := j.countFn(string(qJSON[i]))
		if err != nil {
			return true, nil
		}
		total += n
		if n > longest {
			longest, longestID = n, id
		}
	}

	if total > MaxTotalTokens {
		return true, &llmgate.LLMError{
			Class:    llmgate.ErrClassContextLength,
			Provider: llmgate.ProviderTypeSafe,
			Message: fmt.Sprintf(
				"state plus %d questions is about %d tokens, over the %d limit — send fewer questions per call, or trim the state",
				len(questions), total, MaxTotalTokens),
		}
	}

	if stateTokens+longest > MaxStateAndQuestionTokens {
		return true, &llmgate.LLMError{
			Class:    llmgate.ErrClassContextLength,
			Provider: llmgate.ProviderTypeSafe,
			Message: fmt.Sprintf(
				"state plus the longest question (%q) is about %d tokens, over the %d per-question limit",
				longestID, stateTokens+longest, MaxStateAndQuestionTokens),
		}
	}
	return true, nil
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
//
// The encoder is built once per process. tiktoken-go caches the rank
// table but builds a fresh BPE — a 100k-entry map and a regex compile —
// on every GetEncoding, which measured 256 ms per call; the guard used to
// pay that once per question (HAL-1708).
//
// Past that, encoding itself is the cost: tiktoken-go's pre-tokenizer is a
// backtracking regex, about 4 µs a byte, so a 16-page request state took
// 290 ms to count — on every request, inside the concurrency limiter's
// slot. Long text is therefore cut into pieces and counted in parallel.
// The count is exact, not approximate: see splitForCount.
func EstimateTokens(text string) (int, error) {
	enc, err := cl100k()
	if err != nil {
		return 0, err
	}
	chunks := splitForCount(text, parallelCountBytes)
	if len(chunks) == 1 {
		return len(enc.Encode(text, nil, nil)), nil
	}
	counts := make([]int, len(chunks))
	var wg sync.WaitGroup
	for i, c := range chunks {
		wg.Add(1)
		go func() {
			defer wg.Done()
			counts[i] = len(enc.Encode(c, nil, nil))
		}()
	}
	wg.Wait()
	n := 0
	for _, c := range counts {
		n += c
	}
	return n, nil
}

// parallelCountBytes is the chunk size above which counting fans out.
// Below it a goroutine costs more than it saves.
const parallelCountBytes = 8 << 10

// splitForCount cuts text into chunks of roughly size bytes, only at
// positions where the count of the parts provably equals the count of the
// whole.
//
// cl100k's BPE never merges across a pre-tokenizer piece, so a cut that
// falls on a piece boundary changes nothing. One such boundary is easy to
// recognise without running the regex: a single space, preceded by a
// non-space and followed by a letter. The pattern gives that space to the
// word after it (` ?\p{L}+`), and the piece before it ends at the
// non-space; no alternative in the pattern spans a lone space between a
// non-space and a letter. Cutting just before the space keeps " word"
// whole in the next chunk.
//
// Text with no such position — a single enormous token-dense run — comes
// back as one chunk and is counted serially, which is merely slower.
func splitForCount(text string, size int) []string {
	if len(text) <= size {
		return []string{text}
	}
	var out []string
	for len(text) > size {
		cut := -1
		for i := size; i < len(text)-1 && i < size*2; i++ {
			if text[i] == ' ' && !isSpaceByte(text[i-1]) && isASCIILetter(text[i+1]) {
				cut = i
				break
			}
		}
		if cut < 0 {
			break
		}
		out = append(out, text[:cut])
		text = text[cut:]
	}
	return append(out, text)
}

func isSpaceByte(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\v' || b == '\f'
}

func isASCIILetter(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

var cl100k = sync.OnceValues(func() (*tiktoken.Tiktoken, error) {
	enc, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		return nil, fmt.Errorf("typesafe: load tokenizer: %w", err)
	}
	return enc, nil
})

// Warm opens a connection to the API ahead of the first judgment, so the
// first real call does not pay DNS, TCP and TLS on the critical path, and
// builds the guard's tokenizer so the first oversized request does not pay
// for that either.
//
// It sends an unauthenticated GET to the API root; the status is ignored —
// any response means a connection now sits in the pool. HTTP/2 multiplexes
// later calls onto it. Call it at startup or when a session begins; it is
// never required.
func (j *Judge) Warm(ctx context.Context) error {
	if j.byteBound && !j.skipGuard {
		go func() { _, _ = cl100k() }() // an error resurfaces on first use
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, j.baseURL+"/", nil)
	if err != nil {
		return fmt.Errorf("typesafe: warm: %w", err)
	}
	resp, err := j.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("typesafe: warm: %w", err)
	}
	// Drained so the connection returns to the pool.
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
	return resp.Body.Close()
}

// truncate shortens s to at most n characters, marking that it was cut.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
