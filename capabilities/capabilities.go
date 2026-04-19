// Package capabilities tracks what each known LLM model supports
// (context window, JSON mode, streaming, tools, vision) and provides
// a Capable interface middleware and routers can use to introspect
// an underlying Client's model.
package capabilities

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
// implement it. Middleware, router, and engines can type-assert.
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
var caps = func() map[string]Capabilities {
	m := make(map[string]Capabilities, len(defaultCapabilities))
	for k, v := range defaultCapabilities {
		m[k] = v
	}
	return m
}()

// Lookup returns known caps for a model, zero value if unknown.
func Lookup(model string) Capabilities {
	capMu.RLock()
	defer capMu.RUnlock()
	return caps[model]
}

// Register overrides or adds capabilities for a model.
func Register(model string, c Capabilities) {
	capMu.Lock()
	defer capMu.Unlock()
	caps[model] = c
}

// Of returns the Capabilities of c if it implements Capable, else the
// zero value. Middleware wrappers use this to delegate capability
// lookups transparently to their inner Client.
//
// The parameter is declared as interface{} to avoid importing the root
// llmgate package (which would create an import cycle, since root must
// not depend on any subpackage). Any Client implementation that also
// implements Capable will satisfy the type assertion at runtime.
func Of(c interface{}) Capabilities {
	if cap, ok := c.(Capable); ok {
		return cap.Capabilities()
	}
	return Capabilities{}
}
