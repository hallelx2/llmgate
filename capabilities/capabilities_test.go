package capabilities_test

import (
	"testing"

	"github.com/hallelx2/llmgate/capabilities"
)

func TestLookupKnown(t *testing.T) {
	c := capabilities.Lookup("gemini-2.5-pro")
	if c.MaxContext != 2_000_000 {
		t.Fatalf("gemini-2.5-pro MaxContext = %d, want 2000000", c.MaxContext)
	}
	if !c.SupportsVision || !c.SupportsTools || !c.SupportsJSONMode || !c.SupportsStreaming {
		t.Fatalf("gemini-2.5-pro caps incomplete: %+v", c)
	}
}

func TestLookupUnknown(t *testing.T) {
	c := capabilities.Lookup("nonexistent-zzz")
	if (c != capabilities.Capabilities{}) {
		t.Fatalf("expected zero Capabilities, got %+v", c)
	}
}

func TestRegister(t *testing.T) {
	want := capabilities.Capabilities{MaxContext: 1234, SupportsTools: true}
	capabilities.Register("test-custom-cap-model", want)
	got := capabilities.Lookup("test-custom-cap-model")
	if got != want {
		t.Fatalf("Register round-trip: got %+v want %+v", got, want)
	}
}

func TestO3NoVision(t *testing.T) {
	c := capabilities.Lookup("o3")
	if c.SupportsVision {
		t.Fatalf("o3 should not support vision")
	}
	if !c.SupportsTools {
		t.Fatalf("o3 should support tools")
	}
}
