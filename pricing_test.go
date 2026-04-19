package llmgate_test

import (
	"math"
	"testing"

	"github.com/hallelx2/llmgate"
)

func TestLookupPriceKnown(t *testing.T) {
	p, ok := llmgate.LookupPrice("claude-sonnet-4-5")
	if !ok {
		t.Fatalf("expected claude-sonnet-4-5 to be priced")
	}
	if p.InputPerMTok != 3.00 || p.OutputPerMTok != 15.00 {
		t.Fatalf("unexpected price %+v", p)
	}
}

func TestLookupPriceUnknown(t *testing.T) {
	if _, ok := llmgate.LookupPrice("nonexistent-model-zzz"); ok {
		t.Fatalf("expected unknown model to be unpriced")
	}
}

func TestRegisterPrice(t *testing.T) {
	llmgate.RegisterPrice("test-custom-model", llmgate.Price{InputPerMTok: 1.23, OutputPerMTok: 4.56})
	p, ok := llmgate.LookupPrice("test-custom-model")
	if !ok || p.InputPerMTok != 1.23 || p.OutputPerMTok != 4.56 {
		t.Fatalf("RegisterPrice round-trip failed: %+v ok=%v", p, ok)
	}
}

func TestComputeCostUSD(t *testing.T) {
	// gpt-4o-mini: 0.15 input / 0.60 output per million.
	// 1000 in + 500 out = 0.15 * 0.001 + 0.60 * 0.0005 = 0.00015 + 0.0003 = 0.00045
	got := llmgate.ComputeCostUSD("gpt-4o-mini", 1000, 500)
	want := 0.00045
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("ComputeCostUSD = %v, want %v", got, want)
	}
}

func TestComputeCostUSDUnknown(t *testing.T) {
	if got := llmgate.ComputeCostUSD("nonexistent-model-zzz", 1000, 1000); got != 0 {
		t.Fatalf("expected 0 for unknown model, got %v", got)
	}
}
