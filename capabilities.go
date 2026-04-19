package llmgate

import "sync"

// Capabilities describes what a model supports.
type Capabilities struct {
	MaxContext        int
	SupportsJSONMode  bool
	SupportsStreaming bool
	SupportsTools     bool
	SupportsVision    bool
}

// Capable is optional — Clients that know their model's capabilities
// implement it. Router and engines can type-assert.
type Capable interface {
	Capabilities() Capabilities
}

var defaultCapabilities = map[string]Capabilities{
	// Anthropic Claude 4.x family
	"claude-sonnet-4-5": {MaxContext: 200000, SupportsJSONMode: true, SupportsStreaming: true, SupportsTools: true, SupportsVision: true},
	"claude-opus-4-1":   {MaxContext: 200000, SupportsJSONMode: true, SupportsStreaming: true, SupportsTools: true, SupportsVision: true},
	"claude-haiku-4-5":  {MaxContext: 200000, SupportsJSONMode: true, SupportsStreaming: true, SupportsTools: true, SupportsVision: true},

	// OpenAI
	"gpt-4o":       {MaxContext: 128000, SupportsJSONMode: true, SupportsStreaming: true, SupportsTools: true, SupportsVision: true},
	"gpt-4o-mini":  {MaxContext: 128000, SupportsJSONMode: true, SupportsStreaming: true, SupportsTools: true, SupportsVision: true},
	"gpt-4.1":      {MaxContext: 128000, SupportsJSONMode: true, SupportsStreaming: true, SupportsTools: true, SupportsVision: true},
	"gpt-4.1-mini": {MaxContext: 128000, SupportsJSONMode: true, SupportsStreaming: true, SupportsTools: true, SupportsVision: true},
	"o3":           {MaxContext: 200000, SupportsJSONMode: true, SupportsStreaming: true, SupportsTools: true, SupportsVision: false},
	"o4-mini":      {MaxContext: 200000, SupportsJSONMode: true, SupportsStreaming: true, SupportsTools: true, SupportsVision: false},

	// Google
	"gemini-2.5-pro":   {MaxContext: 2000000, SupportsJSONMode: true, SupportsStreaming: true, SupportsTools: true, SupportsVision: true},
	"gemini-2.5-flash": {MaxContext: 1000000, SupportsJSONMode: true, SupportsStreaming: true, SupportsTools: true, SupportsVision: true},
	"gemini-2.0-flash": {MaxContext: 1000000, SupportsJSONMode: true, SupportsStreaming: true, SupportsTools: true, SupportsVision: true},
}

var capMu sync.RWMutex
var capabilities = func() map[string]Capabilities {
	m := make(map[string]Capabilities, len(defaultCapabilities))
	for k, v := range defaultCapabilities {
		m[k] = v
	}
	return m
}()

// LookupCapabilities returns known caps for a model, zero value if unknown.
func LookupCapabilities(model string) Capabilities {
	capMu.RLock()
	defer capMu.RUnlock()
	return capabilities[model]
}

// RegisterCapabilities overrides or adds capabilities for a model.
func RegisterCapabilities(model string, c Capabilities) {
	capMu.Lock()
	defer capMu.Unlock()
	capabilities[model] = c
}

// capsOf returns the Capabilities of c if it implements Capable, else zero.
// Used by middleware to transparently expose the underlying model's caps.
func capsOf(c Client) Capabilities {
	if cap, ok := c.(Capable); ok {
		return cap.Capabilities()
	}
	return Capabilities{}
}
