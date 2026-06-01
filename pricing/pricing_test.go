package pricing_test

import (
	"math"
	"testing"

	"github.com/hallelx2/llmgate/pricing"
)

func TestLookupKnown(t *testing.T) {
	p, ok := pricing.Lookup("claude-sonnet-4-5")
	if !ok {
		t.Fatalf("expected claude-sonnet-4-5 to be priced")
	}
	if p.InputPerMTok != 3.00 || p.OutputPerMTok != 15.00 {
		t.Fatalf("unexpected price %+v", p)
	}
}

func TestLookupUnknown(t *testing.T) {
	if _, ok := pricing.Lookup("nonexistent-model-zzz"); ok {
		t.Fatalf("expected unknown model to be unpriced")
	}
}

func TestRegister(t *testing.T) {
	pricing.Register("test-custom-model", pricing.Price{InputPerMTok: 1.23, OutputPerMTok: 4.56})
	p, ok := pricing.Lookup("test-custom-model")
	if !ok || p.InputPerMTok != 1.23 || p.OutputPerMTok != 4.56 {
		t.Fatalf("Register round-trip failed: %+v ok=%v", p, ok)
	}
}

func TestCompute(t *testing.T) {
	// gpt-4o-mini: 0.15 input / 0.60 output per million.
	// 1000 in + 500 out = 0.15 * 0.001 + 0.60 * 0.0005 = 0.00015 + 0.0003 = 0.00045
	got := pricing.Compute("gpt-4o-mini", 1000, 500)
	want := 0.00045
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("Compute = %v, want %v", got, want)
	}
}

func TestComputeUnknown(t *testing.T) {
	if got := pricing.Compute("nonexistent-model-zzz", 1000, 1000); got != 0 {
		t.Fatalf("expected 0 for unknown model, got %v", got)
	}
}

func TestComputeWithOK(t *testing.T) {
	cost, ok := pricing.ComputeWithOK("gpt-4o-mini", 1000, 500)
	if !ok {
		t.Fatalf("expected gpt-4o-mini to be priced")
	}
	if math.Abs(cost-0.00045) > 1e-9 {
		t.Fatalf("ComputeWithOK cost = %v, want 0.00045", cost)
	}

	// Unknown model: cost 0 AND priced=false, so callers can tell an
	// unpriced call apart from a genuinely free one.
	cost, ok = pricing.ComputeWithOK("nonexistent-model-zzz-2", 1000, 1000)
	if ok {
		t.Fatalf("expected unknown model to report priced=false")
	}
	if cost != 0 {
		t.Fatalf("expected 0 cost for unknown model, got %v", cost)
	}
}

func TestGLMPriced(t *testing.T) {
	p, ok := pricing.Lookup("glm-4.6")
	if !ok {
		t.Fatalf("glm-4.6 must be in the price book (this was the benchmark $0 regression)")
	}
	if p.InputPerMTok <= 0 || p.OutputPerMTok <= 0 {
		t.Fatalf("glm-4.6 prices must be positive, got %+v", p)
	}
}

func TestWarnFuncFiresOncePerModel(t *testing.T) {
	var calls []string
	pricing.WarnFunc = func(model string) { calls = append(calls, model) }
	t.Cleanup(func() { pricing.WarnFunc = nil })

	const model = "unpriced-warn-test-model"
	for i := 0; i < 3; i++ {
		pricing.ComputeWithOK(model, 10, 10)
	}
	// A known model must never warn.
	pricing.ComputeWithOK("gpt-4o-mini", 10, 10)

	if len(calls) != 1 || calls[0] != model {
		t.Fatalf("WarnFunc should fire exactly once for the unpriced model, got %v", calls)
	}
}
