package adapter

import (
	"strings"
	"testing"

	"github.com/tmc/langchaingo/llms"

	"github.com/hallelx2/llmgate"
)

// cachedParts reports how many parts of a message carry a cache marker.
func cachedParts(m llms.MessageContent) int {
	n := 0
	for _, p := range m.Parts {
		if _, ok := p.(llms.CachedContent); ok {
			n++
		}
	}
	return n
}

func TestCacheBreakpointMarksUserMessage(t *testing.T) {
	msgs := toLangchainMessages([]llmgate.Message{
		{Role: llmgate.RoleUser, Content: "a long stable document", CacheBreakpoint: true},
		{Role: llmgate.RoleUser, Content: "the question"},
	}, false, nil)

	if got := cachedParts(msgs[0]); got != 1 {
		t.Errorf("first message has %d cached parts, want 1", got)
	}
	if got := cachedParts(msgs[1]); got != 0 {
		t.Errorf("unmarked message has %d cached parts, want 0", got)
	}

	cc, ok := msgs[0].Parts[0].(llms.CachedContent)
	if !ok {
		t.Fatalf("part is %T, want llms.CachedContent", msgs[0].Parts[0])
	}
	if cc.CacheControl == nil || cc.CacheControl.Type != "ephemeral" {
		t.Errorf("cache control = %+v, want type ephemeral", cc.CacheControl)
	}
	if tc, ok := cc.ContentPart.(llms.TextContent); !ok || tc.Text != "a long stable document" {
		t.Errorf("wrapped content = %+v, want the original text intact", cc.ContentPart)
	}
}

// TestCacheBreakpointIgnoredOnSystemMessage guards a failure mode that would
// not degrade gracefully. langchaingo's Anthropic handler accepts a bare
// TextContent for the system prompt and returns ErrInvalidContentType for
// anything else, so marking a system message would not cache it — it would
// fail every request.
func TestCacheBreakpointIgnoredOnSystemMessage(t *testing.T) {
	for _, role := range []llmgate.Role{llmgate.RoleSystem, llmgate.RoleAssistant, llmgate.RoleTool} {
		msgs := toLangchainMessages([]llmgate.Message{
			{Role: role, Content: "content", CacheBreakpoint: true, ToolCallID: "call_1"},
		}, false, nil)
		if got := cachedParts(msgs[0]); got != 0 {
			t.Errorf("role %v: %d cached parts, want 0 — only user turns may carry a cache marker", role, got)
		}
	}
}

func TestPromptCacheShorthandMarksFirstUserMessage(t *testing.T) {
	in := []llmgate.Message{
		{Role: llmgate.RoleSystem, Content: "you are a helpful assistant"},
		{Role: llmgate.RoleUser, Content: "a long stable document"},
		{Role: llmgate.RoleAssistant, Content: "understood"},
		{Role: llmgate.RoleUser, Content: "the question"},
	}

	got := withPromptCache(in, true)

	if !got[1].CacheBreakpoint {
		t.Error("the first user message should carry the breakpoint")
	}
	if got[3].CacheBreakpoint {
		t.Error("the trailing user message must not be cached — it changes every call")
	}
	if got[0].CacheBreakpoint {
		t.Error("the system message must not be marked")
	}
}

// TestPromptCacheShorthandDefersToCaller: an explicit breakpoint anywhere
// means the caller has chosen the prefix, and adding a second one would
// cache something they did not pick — providers also cap how many
// breakpoints a request may carry.
func TestPromptCacheShorthandDefersToCaller(t *testing.T) {
	in := []llmgate.Message{
		{Role: llmgate.RoleUser, Content: "part one"},
		{Role: llmgate.RoleUser, Content: "part two", CacheBreakpoint: true},
	}

	got := withPromptCache(in, true)

	if got[0].CacheBreakpoint {
		t.Error("shorthand overrode an explicit caller breakpoint")
	}
	if !got[1].CacheBreakpoint {
		t.Error("the caller's breakpoint was lost")
	}
}

// TestPromptCacheDoesNotMutateCallerSlice matters because Request.Messages
// belongs to the caller and a retry or middleware chain may hand the same
// backing array back on the next attempt.
func TestPromptCacheDoesNotMutateCallerSlice(t *testing.T) {
	in := []llmgate.Message{{Role: llmgate.RoleUser, Content: "document"}}

	got := withPromptCache(in, true)

	if in[0].CacheBreakpoint {
		t.Error("withPromptCache mutated the caller's slice")
	}
	if !got[0].CacheBreakpoint {
		t.Error("the returned copy is missing the breakpoint")
	}
}

func TestPromptCacheOffChangesNothing(t *testing.T) {
	in := []llmgate.Message{{Role: llmgate.RoleUser, Content: "document"}}
	if withPromptCache(in, false)[0].CacheBreakpoint {
		t.Error("shorthand applied while disabled")
	}
}

// TestJSONNudgeSurvivesACachedTail is the interaction the wrapper could
// easily have broken: the nudge is appended by type-asserting the last part
// to TextContent, and a cached part is not one. Losing it silently would
// turn JSON mode into a no-op.
//
// The nudge also has to land *outside* the cached block. It carries a
// per-call schema, and changing the tail of a cached prefix invalidates the
// cache on every request.
func TestJSONNudgeSurvivesACachedTail(t *testing.T) {
	msgs := toLangchainMessages([]llmgate.Message{
		{Role: llmgate.RoleUser, Content: "the document", CacheBreakpoint: true},
	}, true, []byte(`{"type":"object"}`))

	last := msgs[len(msgs)-1]
	var found bool
	for _, p := range last.Parts {
		if tc, ok := p.(llms.TextContent); ok && strings.Contains(tc.Text, "single JSON object") {
			found = true
		}
	}
	if !found {
		t.Fatalf("the JSON nudge was dropped; parts = %+v", last.Parts)
	}

	if cachedParts(last) != 1 {
		t.Errorf("%d cached parts, want exactly 1 — the nudge must stay outside the cached block",
			cachedParts(last))
	}
	if cc, ok := last.Parts[0].(llms.CachedContent); ok {
		if tc, ok := cc.ContentPart.(llms.TextContent); ok && strings.Contains(tc.Text, "single JSON object") {
			t.Error("the nudge was appended inside the cached block, which invalidates the cache every call")
		}
	}
}
