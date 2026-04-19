package llmgate_test

import (
	"testing"

	"github.com/hallelx2/llmgate"
)

func TestLookupCapabilitiesKnown(t *testing.T) {
	c := llmgate.LookupCapabilities("gemini-2.5-pro")
	if c.MaxContext != 2_000_000 {
		t.Fatalf("gemini-2.5-pro MaxContext = %d, want 2000000", c.MaxContext)
	}
	if !c.SupportsVision || !c.SupportsTools || !c.SupportsJSONMode || !c.SupportsStreaming {
		t.Fatalf("gemini-2.5-pro caps incomplete: %+v", c)
	}
}

func TestLookupCapabilitiesUnknown(t *testing.T) {
	c := llmgate.LookupCapabilities("nonexistent-zzz")
	if (c != llmgate.Capabilities{}) {
		t.Fatalf("expected zero Capabilities, got %+v", c)
	}
}

func TestRegisterCapabilities(t *testing.T) {
	want := llmgate.Capabilities{MaxContext: 1234, SupportsTools: true}
	llmgate.RegisterCapabilities("test-custom-cap-model", want)
	got := llmgate.LookupCapabilities("test-custom-cap-model")
	if got != want {
		t.Fatalf("RegisterCapabilities round-trip: got %+v want %+v", got, want)
	}
}

func TestO3NoVision(t *testing.T) {
	c := llmgate.LookupCapabilities("o3")
	if c.SupportsVision {
		t.Fatalf("o3 should not support vision")
	}
	if !c.SupportsTools {
		t.Fatalf("o3 should support tools")
	}
}
