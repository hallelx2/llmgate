package vertex

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/hallelx2/llmgate"
)

// Wire types for the Vertex generateContent API (v1). Only the fields
// llmgate reads or writes are modelled.

type part struct {
	Text             string            `json:"text,omitempty"`
	Thought          bool              `json:"thought,omitempty"`
	ThoughtSignature string            `json:"thoughtSignature,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
}

type functionCall struct {
	ID   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type functionResponse struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type content struct {
	Role  string `json:"role,omitempty"`
	Parts []part `json:"parts"`
}

type thinkingConfig struct {
	ThinkingBudget  *int   `json:"thinkingBudget,omitempty"`
	ThinkingLevel   string `json:"thinkingLevel,omitempty"`
	IncludeThoughts bool   `json:"includeThoughts,omitempty"`
}

type generationConfig struct {
	MaxOutputTokens    int             `json:"maxOutputTokens,omitempty"`
	Temperature        *float64        `json:"temperature,omitempty"`
	TopP               *float64        `json:"topP,omitempty"`
	Seed               *int            `json:"seed,omitempty"`
	ResponseMimeType   string          `json:"responseMimeType,omitempty"`
	ResponseJSONSchema json.RawMessage `json:"responseJsonSchema,omitempty"`
	ThinkingConfig     *thinkingConfig `json:"thinkingConfig,omitempty"`
}

type functionDeclaration struct {
	Name                 string          `json:"name"`
	Description          string          `json:"description,omitempty"`
	ParametersJSONSchema json.RawMessage `json:"parametersJsonSchema,omitempty"`
}

type tool struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations"`
}

type functionCallingConfig struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type toolConfig struct {
	FunctionCallingConfig functionCallingConfig `json:"functionCallingConfig"`
}

type generateRequest struct {
	Contents          []content         `json:"contents"`
	SystemInstruction *content          `json:"systemInstruction,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
	Tools             []tool            `json:"tools,omitempty"`
	ToolConfig        *toolConfig       `json:"toolConfig,omitempty"`
}

type usageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	ToolUsePromptTokenCount int `json:"toolUsePromptTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
}

type candidate struct {
	Content      content `json:"content"`
	FinishReason string  `json:"finishReason"`
}

type generateResponse struct {
	Candidates     []candidate    `json:"candidates"`
	UsageMetadata  *usageMetadata `json:"usageMetadata"`
	ModelVersion   string         `json:"modelVersion"`
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
}

// buildRequest maps an llmgate.Request onto generateContent.
//
// System messages become systemInstruction (Vertex has no system role in
// contents). Assistant turns become role "model". Tool results become a
// user turn of functionResponse parts; consecutive tool results are
// grouped into one turn, since Vertex expects every response to a set of
// parallel calls in the turn that follows them.
func buildRequest(req llmgate.Request, th Thinking) (*generateRequest, error) {
	out := &generateRequest{}

	var system []string
	// Tool names by call ID: a functionResponse must name its function,
	// and llmgate's tool message carries only the ID.
	names := map[string]string{}

	for _, m := range req.Messages {
		switch m.Role {
		case llmgate.RoleSystem:
			if m.Content != "" {
				system = append(system, m.Content)
			}

		case llmgate.RoleUser:
			out.Contents = append(out.Contents, content{Role: "user", Parts: []part{{Text: m.Content}}})

		case llmgate.RoleAssistant:
			c := content{Role: "model"}
			if m.Content != "" {
				c.Parts = append(c.Parts, part{Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				id, sig := decodeCallID(tc.ID)
				names[tc.ID] = tc.Name
				args := json.RawMessage(tc.Input)
				if len(args) == 0 {
					args = json.RawMessage("{}")
				}
				if !json.Valid(args) {
					return nil, fmt.Errorf("llmgate/vertex: tool call %q has invalid JSON input", tc.Name)
				}
				c.Parts = append(c.Parts, part{
					FunctionCall:     &functionCall{ID: id, Name: tc.Name, Args: args},
					ThoughtSignature: sig,
				})
			}
			if len(c.Parts) == 0 {
				c.Parts = []part{{Text: ""}}
			}
			out.Contents = append(out.Contents, c)

		case llmgate.RoleTool:
			name := names[m.ToolCallID]
			if name == "" {
				return nil, fmt.Errorf("llmgate/vertex: tool result %q answers no earlier tool call", m.ToolCallID)
			}
			id, _ := decodeCallID(m.ToolCallID)
			fr := part{FunctionResponse: &functionResponse{ID: id, Name: name, Response: toolResult(m.Content)}}
			if n := len(out.Contents); n > 0 && isToolTurn(out.Contents[n-1]) {
				out.Contents[n-1].Parts = append(out.Contents[n-1].Parts, fr)
			} else {
				out.Contents = append(out.Contents, content{Role: "user", Parts: []part{fr}})
			}

		default:
			return nil, fmt.Errorf("llmgate/vertex: unsupported role %q", m.Role)
		}
	}
	if len(out.Contents) == 0 {
		return nil, fmt.Errorf("llmgate/vertex: request has no user or assistant messages")
	}
	if len(system) > 0 {
		out.SystemInstruction = &content{Parts: []part{{Text: strings.Join(system, "\n\n")}}}
	}

	gc := &generationConfig{
		MaxOutputTokens: req.MaxTokens,
		Temperature:     req.Temperature,
		TopP:            req.TopP,
		Seed:            req.Seed,
	}
	if req.JSONMode {
		gc.ResponseMimeType = "application/json"
		if len(req.JSONSchema) > 0 {
			if !json.Valid(req.JSONSchema) {
				return nil, fmt.Errorf("llmgate/vertex: JSONSchema is not valid JSON")
			}
			gc.ResponseJSONSchema = json.RawMessage(req.JSONSchema)
		}
	}
	if !th.isZero() {
		gc.ThinkingConfig = &thinkingConfig{ThinkingBudget: th.Budget, ThinkingLevel: th.Level, IncludeThoughts: th.IncludeThoughts}
	}
	if gc.MaxOutputTokens != 0 || gc.Temperature != nil || gc.TopP != nil || gc.Seed != nil ||
		gc.ResponseMimeType != "" || gc.ThinkingConfig != nil {
		out.GenerationConfig = gc
	}

	if len(req.Tools) > 0 {
		decls := make([]functionDeclaration, 0, len(req.Tools))
		for _, t := range req.Tools {
			d := functionDeclaration{Name: t.Name, Description: t.Description}
			if len(t.InputSchema) > 0 {
				if !json.Valid(t.InputSchema) {
					return nil, fmt.Errorf("llmgate/vertex: tool %q has an invalid input schema", t.Name)
				}
				d.ParametersJSONSchema = json.RawMessage(t.InputSchema)
			}
			decls = append(decls, d)
		}
		out.Tools = []tool{{FunctionDeclarations: decls}}

		switch choice := req.ToolChoice; choice {
		case "", "auto":
			// Vertex's default is AUTO.
		case "required", "any":
			out.ToolConfig = &toolConfig{FunctionCallingConfig: functionCallingConfig{Mode: "ANY"}}
		case "none":
			out.ToolConfig = &toolConfig{FunctionCallingConfig: functionCallingConfig{Mode: "NONE"}}
		default:
			out.ToolConfig = &toolConfig{FunctionCallingConfig: functionCallingConfig{Mode: "ANY", AllowedFunctionNames: []string{choice}}}
		}
	}
	return out, nil
}

func isToolTurn(c content) bool {
	if c.Role != "user" || len(c.Parts) == 0 {
		return false
	}
	for _, p := range c.Parts {
		if p.FunctionResponse == nil {
			return false
		}
	}
	return true
}

// toolResult wraps a tool's output as the object functionResponse
// requires. A JSON object passes through; anything else is sent as
// {"output": <text>}.
func toolResult(s string) map[string]any {
	var obj map[string]any
	if err := json.Unmarshal([]byte(s), &obj); err == nil && obj != nil {
		return obj
	}
	return map[string]any{"output": s}
}

// Call IDs carry Gemini 3's thoughtSignature across the round trip:
//
//	vtx1.<vertex call id>.<response nonce>.<index>.<base64url signature>
//
// The signature is already base64 from Vertex; re-encoding as base64url
// keeps the ID free of '/' and '+'. The nonce is per response, so calls in
// different responses never share an ID even when Vertex sends neither an
// ID nor a signature. An ID that does not start with the
// prefix (one a caller made up) decodes to itself with no signature.
const callIDPrefix = "vtx1."

func encodeCallID(vertexID, nonce string, index int, signature string) string {
	return callIDPrefix + base64.RawURLEncoding.EncodeToString([]byte(vertexID)) + "." + nonce + "." +
		strconv.Itoa(index) + "." + base64.RawURLEncoding.EncodeToString([]byte(signature))
}

func decodeCallID(id string) (vertexID, signature string) {
	rest, ok := strings.CutPrefix(id, callIDPrefix)
	if !ok {
		return id, ""
	}
	fields := strings.SplitN(rest, ".", 4)
	if len(fields) != 4 {
		return id, ""
	}
	v, err1 := base64.RawURLEncoding.DecodeString(fields[0])
	s, err2 := base64.RawURLEncoding.DecodeString(fields[3])
	if err1 != nil || err2 != nil {
		return id, ""
	}
	return string(v), string(s)
}

// httpError turns a non-200 response into a classified *llmgate.LLMError.
func httpError(resp *http.Response, body []byte) error {
	msg, status := errorMessage(body)
	e := llmgate.NewLLMError(llmgate.ProviderGemini, resp.StatusCode, "vertex: "+msg, nil)
	lower := strings.ToLower(msg)
	switch {
	case resp.StatusCode == http.StatusTooManyRequests || status == "RESOURCE_EXHAUSTED":
		e.Class = llmgate.ErrClassRateLimited
		e.RetryAfterDur = retryAfter(resp.Header)
	case resp.StatusCode == http.StatusBadRequest &&
		(strings.Contains(lower, "exceeds the maximum number of tokens") || strings.Contains(lower, "input token count")):
		e.Class = llmgate.ErrClassContextLength
	case resp.StatusCode == http.StatusNotFound:
		// An unknown model or a model not enabled in this project: a
		// configuration fault, not something a retry fixes.
		e.Class = llmgate.ErrClassBadRequest
	case status == "DEADLINE_EXCEEDED" || resp.StatusCode == http.StatusGatewayTimeout:
		e.Class = llmgate.ErrClassTimeout
	}
	return e
}

func errorMessage(body []byte) (msg, status string) {
	var wire struct {
		Error struct {
			Message string `json:"message"`
			Status  string `json:"status"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &wire); err == nil && wire.Error.Message != "" {
		return wire.Error.Message, wire.Error.Status
	}
	s := strings.TrimSpace(string(body))
	if len(s) > 300 {
		s = s[:300] + "…"
	}
	return s, ""
}

func retryAfter(h http.Header) time.Duration {
	v := strings.TrimSpace(h.Get("Retry-After"))
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil && secs > 0 {
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}
