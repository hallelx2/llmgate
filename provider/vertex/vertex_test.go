package vertex

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"golang.org/x/oauth2"

	"github.com/hallelx2/llmgate"
)

// fakeVertex records the last request and answers with a canned body.
type fakeVertex struct {
	status  int
	body    string
	headers map[string]string

	gotPath   string
	gotAuth   string
	gotProj   string
	gotBody   map[string]any
	gotRawReq []byte
}

func (f *fakeVertex) handler(w http.ResponseWriter, r *http.Request) {
	f.gotPath = r.URL.Path
	f.gotAuth = r.Header.Get("Authorization")
	f.gotProj = r.Header.Get("x-goog-user-project")
	f.gotRawReq, _ = io.ReadAll(r.Body)
	f.gotBody = map[string]any{}
	_ = json.Unmarshal(f.gotRawReq, &f.gotBody)
	for k, v := range f.headers {
		w.Header().Set(k, v)
	}
	if f.status == 0 {
		f.status = http.StatusOK
	}
	w.WriteHeader(f.status)
	_, _ = io.WriteString(w, f.body)
}

func newTestClient(t *testing.T, f *fakeVertex, cfg Config) *Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(f.handler))
	t.Cleanup(srv.Close)
	cfg.BaseURL = srv.URL
	cfg.TokenSource = oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "tok-123"})
	if cfg.Project == "" {
		cfg.Project = "proj-x"
	}
	c, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

const okBody = `{
  "candidates": [{"content": {"role": "model", "parts": [{"text": "391", "thoughtSignature": "c2ln"}]}, "finishReason": "STOP"}],
  "usageMetadata": {"promptTokenCount": 1000, "cachedContentTokenCount": 400, "candidatesTokenCount": 20, "thoughtsTokenCount": 56, "totalTokenCount": 1076},
  "modelVersion": "gemini-2.5-flash-lite"
}`

func TestCompleteRequestShapeAndAuth(t *testing.T) {
	f := &fakeVertex{body: okBody}
	c := newTestClient(t, f, Config{Location: "global", Model: "gemini-2.5-flash-lite", Thinking: Thinking{Budget: llmgate.Int(0)}})

	_, err := c.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{
			{Role: llmgate.RoleSystem, Content: "be brief"},
			{Role: llmgate.RoleUser, Content: "17*23?"},
		},
		MaxTokens:   64,
		Temperature: llmgate.Float64(0),
		Seed:        llmgate.Int(7),
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if want := "/v1/projects/proj-x/locations/global/publishers/google/models/gemini-2.5-flash-lite:generateContent"; f.gotPath != want {
		t.Errorf("path = %q, want %q", f.gotPath, want)
	}
	if f.gotAuth != "Bearer tok-123" {
		t.Errorf("Authorization = %q", f.gotAuth)
	}
	if f.gotProj != "proj-x" {
		t.Errorf("x-goog-user-project = %q, want proj-x", f.gotProj)
	}
	sys := f.gotBody["systemInstruction"].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"]
	if sys != "be brief" {
		t.Errorf("systemInstruction = %v", sys)
	}
	contents := f.gotBody["contents"].([]any)
	if len(contents) != 1 || contents[0].(map[string]any)["role"] != "user" {
		t.Errorf("contents = %v, want one user turn (system goes to systemInstruction)", contents)
	}
	gc := f.gotBody["generationConfig"].(map[string]any)
	if gc["maxOutputTokens"] != float64(64) || gc["temperature"] != float64(0) || gc["seed"] != float64(7) {
		t.Errorf("generationConfig = %v", gc)
	}
	// Budget 0 must be sent, not dropped as a zero value: it is the
	// difference between thinking and not.
	if !strings.Contains(string(f.gotRawReq), `"thinkingBudget":0`) {
		t.Errorf("request lacks thinkingBudget:0: %s", f.gotRawReq)
	}
}

func TestUsageIsNormalisedAndPriced(t *testing.T) {
	f := &fakeVertex{body: okBody}
	c := newTestClient(t, f, Config{Model: "gemini-2.5-flash-lite"})
	resp, err := c.Complete(context.Background(), llmgate.Request{Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	u := resp.Usage
	// Vertex reports cached tokens inside promptTokenCount; llmgate's
	// Input excludes them.
	if u.InputTokens != 600 || u.CacheReadTokens != 400 {
		t.Errorf("Input/CacheRead = %d/%d, want 600/400", u.InputTokens, u.CacheReadTokens)
	}
	// Thinking is billed as output and reported as its subset.
	if u.OutputTokens != 76 || u.ReasoningTokens != 56 {
		t.Errorf("Output/Reasoning = %d/%d, want 76/56", u.OutputTokens, u.ReasoningTokens)
	}
	if !u.TokensReported || !u.Priced {
		t.Errorf("TokensReported=%v Priced=%v, want both true", u.TokensReported, u.Priced)
	}
	// gemini-2.5-flash-lite: $0.10 in, $0.40 out, $0.01 cached per 1M.
	want := (600*0.10 + 76*0.40 + 400*0.01) / 1e6
	if math.Abs(u.CostUSD-want) > 1e-12 {
		t.Errorf("CostUSD = %.10f, want %.10f", u.CostUSD, want)
	}
	if resp.Content != "391" || resp.FinishReason != "STOP" {
		t.Errorf("Content/Finish = %q/%q", resp.Content, resp.FinishReason)
	}
}

func TestRegionalEndpointAndUplift(t *testing.T) {
	if got := endpointRoot("us-central1"); got != "https://us-central1-aiplatform.googleapis.com" {
		t.Errorf("endpointRoot(us-central1) = %q", got)
	}
	if got := endpointRoot("global"); got != "https://aiplatform.googleapis.com" {
		t.Errorf("endpointRoot(global) = %q", got)
	}
	if regionalUplift("gemini-3.5-flash-lite", "us-central1") != 1.1 {
		t.Error("Gemini 3 off the global endpoint lists 10% higher")
	}
	if regionalUplift("gemini-2.5-flash-lite", "us-central1") != 1 {
		t.Error("Gemini 2.5 has one price in every location")
	}
	if regionalUplift("gemini-3.5-flash-lite", "global") != 1 {
		t.Error("global is the base price")
	}
}

func TestToolCallRoundTripCarriesThoughtSignature(t *testing.T) {
	f := &fakeVertex{body: `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"grep","args":{"pattern":"revenue"},"id":"call_9"},"thoughtSignature":"U0lHK2FiYy8="}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`}
	c := newTestClient(t, f, Config{Model: "gemini-3.5-flash-lite"})

	tools := []llmgate.ToolDef{{Name: "grep", Description: "search", InputSchema: []byte(`{"type":"object","properties":{"pattern":{"type":"string"}}}`)}}
	resp, err := c.Complete(context.Background(), llmgate.Request{
		Messages:   []llmgate.Message{{Role: llmgate.RoleUser, Content: "find revenue"}},
		Tools:      tools,
		ToolChoice: "required",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "grep" || string(resp.ToolCalls[0].Input) != `{"pattern":"revenue"}` {
		t.Fatalf("ToolCalls = %+v", resp.ToolCalls)
	}
	if tc := f.gotBody["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any); tc["mode"] != "ANY" {
		t.Errorf("toolConfig = %v, want mode ANY for required", tc)
	}
	decl := f.gotBody["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
	if decl["name"] != "grep" || decl["parametersJsonSchema"] == nil {
		t.Errorf("functionDeclarations[0] = %v", decl)
	}

	// Second turn: echo the call and answer it. The signature and the
	// Vertex call id must come back on the functionCall part.
	call := resp.ToolCalls[0]
	f.body = okBody
	_, err = c.Complete(context.Background(), llmgate.Request{
		Messages: []llmgate.Message{
			{Role: llmgate.RoleUser, Content: "find revenue"},
			{Role: llmgate.RoleAssistant, ToolCalls: []llmgate.ToolCall{call}},
			{Role: llmgate.RoleTool, ToolCallID: call.ID, Content: "p.41: Revenue was $9.1bn"},
		},
		Tools: tools,
	})
	if err != nil {
		t.Fatal(err)
	}
	contents := f.gotBody["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents has %d turns, want 3", len(contents))
	}
	modelPart := contents[1].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if modelPart["thoughtSignature"] != "U0lHK2FiYy8=" {
		t.Errorf("thoughtSignature echoed = %v, want the original", modelPart["thoughtSignature"])
	}
	if fc := modelPart["functionCall"].(map[string]any); fc["id"] != "call_9" || fc["name"] != "grep" {
		t.Errorf("functionCall echoed = %v", fc)
	}
	toolTurn := contents[2].(map[string]any)
	fr := toolTurn["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if toolTurn["role"] != "user" || fr["name"] != "grep" || fr["id"] != "call_9" {
		t.Errorf("tool turn = %v", toolTurn)
	}
	if fr["response"].(map[string]any)["output"] != "p.41: Revenue was $9.1bn" {
		t.Errorf("functionResponse.response = %v", fr["response"])
	}
}

func TestParallelToolResultsShareOneTurn(t *testing.T) {
	req := llmgate.Request{Messages: []llmgate.Message{
		{Role: llmgate.RoleUser, Content: "q"},
		{Role: llmgate.RoleAssistant, ToolCalls: []llmgate.ToolCall{{ID: "a", Name: "read"}, {ID: "b", Name: "read"}}},
		{Role: llmgate.RoleTool, ToolCallID: "a", Content: `{"page":1}`},
		{Role: llmgate.RoleTool, ToolCallID: "b", Content: `{"page":2}`},
	}}
	got, err := buildRequest(req, Thinking{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Contents) != 3 || len(got.Contents[2].Parts) != 2 {
		t.Fatalf("want 3 turns with both responses in the last; got %+v", got.Contents)
	}
	if got.Contents[2].Parts[0].FunctionResponse.Response["page"] != float64(1) {
		t.Errorf("a JSON object result passes through as the response object")
	}
}

func TestToolResultWithoutCallIsRejected(t *testing.T) {
	_, err := buildRequest(llmgate.Request{Messages: []llmgate.Message{
		{Role: llmgate.RoleUser, Content: "q"},
		{Role: llmgate.RoleTool, ToolCallID: "ghost", Content: "x"},
	}}, Thinking{})
	if err == nil {
		t.Fatal("want an error for a tool result that answers no call")
	}
}

func TestJSONModeSendsSchema(t *testing.T) {
	got, err := buildRequest(llmgate.Request{
		Messages:   []llmgate.Message{{Role: llmgate.RoleUser, Content: "Ada is 36"}},
		JSONMode:   true,
		JSONSchema: []byte(`{"type":"object","properties":{"age":{"type":"integer"}}}`),
	}, Thinking{})
	if err != nil {
		t.Fatal(err)
	}
	if got.GenerationConfig == nil || got.GenerationConfig.ResponseMimeType != "application/json" || got.GenerationConfig.ResponseJSONSchema == nil {
		t.Fatalf("generationConfig = %+v", got.GenerationConfig)
	}
}

func TestNoGenerationConfigWhenNothingSet(t *testing.T) {
	got, err := buildRequest(llmgate.Request{Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "x"}}}, Thinking{})
	if err != nil {
		t.Fatal(err)
	}
	if got.GenerationConfig != nil {
		t.Errorf("generationConfig = %+v, want omitted so Vertex defaults apply", got.GenerationConfig)
	}
}

func TestErrorsAreClassified(t *testing.T) {
	cases := []struct {
		name    string
		status  int
		body    string
		headers map[string]string
		class   llmgate.ErrorClass
	}{
		{"rate limited", 429, `{"error":{"code":429,"message":"Resource exhausted","status":"RESOURCE_EXHAUSTED"}}`, map[string]string{"Retry-After": "3"}, llmgate.ErrClassRateLimited},
		{"auth", 403, `{"error":{"code":403,"message":"Permission denied","status":"PERMISSION_DENIED"}}`, nil, llmgate.ErrClassAuth},
		{"unknown model", 404, `{"error":{"code":404,"message":"Publisher Model was not found","status":"NOT_FOUND"}}`, nil, llmgate.ErrClassBadRequest},
		{"too long", 400, `{"error":{"code":400,"message":"The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).","status":"INVALID_ARGUMENT"}}`, nil, llmgate.ErrClassContextLength},
		{"server", 503, `{"error":{"code":503,"message":"unavailable","status":"UNAVAILABLE"}}`, nil, llmgate.ErrClassTransient},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := &fakeVertex{status: tc.status, body: tc.body, headers: tc.headers}
			c := newTestClient(t, f, Config{})
			_, err := c.Complete(context.Background(), llmgate.Request{Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "x"}}})
			if got := llmgate.Classify(err); got != tc.class {
				t.Errorf("Classify = %v, want %v (err %v)", got, tc.class, err)
			}
			if tc.status == 429 {
				if d, ok := llmgate.RetryAfter(err); !ok || d.Seconds() != 3 {
					t.Errorf("RetryAfter = %v %v, want 3s", d, ok)
				}
			}
		})
	}
}

func TestBlockedPromptIsAContentError(t *testing.T) {
	f := &fakeVertex{body: `{"promptFeedback":{"blockReason":"SAFETY"}}`}
	c := newTestClient(t, f, Config{})
	_, err := c.Complete(context.Background(), llmgate.Request{Messages: []llmgate.Message{{Role: llmgate.RoleUser, Content: "x"}}})
	if llmgate.Classify(err) != llmgate.ErrClassContent {
		t.Errorf("Classify = %v, want content", llmgate.Classify(err))
	}
}

func TestCountTokensUsesEndpoint(t *testing.T) {
	f := &fakeVertex{body: `{"totalTokens": 8}`}
	c := newTestClient(t, f, Config{Model: "gemini-3.5-flash-lite"})
	n, err := c.CountTokens(context.Background(), "hello world, this is a token count")
	if err != nil || n != 8 {
		t.Fatalf("CountTokens = %d, %v", n, err)
	}
	if !strings.HasSuffix(f.gotPath, "/models/gemini-3.5-flash-lite:countTokens") {
		t.Errorf("path = %q", f.gotPath)
	}
}

func TestConfigValidation(t *testing.T) {
	ts := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "x"})
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	if _, err := New(context.Background(), Config{TokenSource: ts}); err == nil {
		t.Error("want an error when no project is known")
	}
	if _, err := New(context.Background(), Config{TokenSource: ts, Project: "p", Thinking: Thinking{Budget: llmgate.Int(0), Level: "low"}}); err == nil {
		t.Error("want an error when both Budget and Level are set")
	}
	t.Setenv("GOOGLE_CLOUD_PROJECT", "from-env")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
	c, err := New(context.Background(), Config{TokenSource: ts})
	if err != nil {
		t.Fatal(err)
	}
	if c.Project() != "from-env" || c.Location() != "europe-west4" || c.Model() != DefaultModel {
		t.Errorf("project/location/model = %s/%s/%s", c.Project(), c.Location(), c.Model())
	}
}

func TestCallIDCodec(t *testing.T) {
	id := encodeCallID("call_1", 2, "a/b+c==")
	if strings.ContainsAny(id, "/+") {
		t.Errorf("encoded id %q should be URL-safe", id)
	}
	v, s := decodeCallID(id)
	if v != "call_1" || s != "a/b+c==" {
		t.Errorf("decode = %q, %q", v, s)
	}
	if v, s := decodeCallID("caller-made-id"); v != "caller-made-id" || s != "" {
		t.Errorf("a foreign id should pass through unchanged, got %q %q", v, s)
	}
}
