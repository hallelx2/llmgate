package pricing

import (
	"math"
	"testing"
)

func TestJevIsPriced(t *testing.T) {
	p, ok := Lookup("jev-1.13.0")
	if !ok {
		t.Fatal("Lookup(jev-1.13.0) = not found; System One models must price")
	}
	if p.InputPerMTok != 0.042 {
		t.Errorf("InputPerMTok = %v, want 0.042", p.InputPerMTok)
	}
	if p.OutputPerMTok != 0 {
		t.Errorf("OutputPerMTok = %v, want 0 — TypeSafe serves output free", p.OutputPerMTok)
	}
}

func TestJevAliasesPrice(t *testing.T) {
	for _, alias := range []string{"jev-latest", "jev-preview"} {
		p, ok := Lookup(alias)
		if !ok {
			t.Errorf("Lookup(%q) = not found", alias)
			continue
		}
		if p.InputPerMTok != 0.042 {
			t.Errorf("Lookup(%q).InputPerMTok = %v, want 0.042", alias, p.InputPerMTok)
		}
	}
}

// A known token count has to produce the expected dollar figure, or the
// cost column in any benchmark is decoration.
func TestJevCostForAKnownTokenCount(t *testing.T) {
	// 1,000,000 input tokens at $0.042/Mtok is exactly $0.042.
	got, ok := ComputeWithOK("jev-1.13.0", 1_000_000, 0)
	if !ok {
		t.Fatal("ComputeWithOK reported the model as unpriced")
	}
	if math.Abs(got-0.042) > 1e-9 {
		t.Errorf("cost for 1M input tokens = %v, want 0.042", got)
	}

	// Output is free, so adding output tokens must not change the bill.
	withOutput, ok := ComputeWithOK("jev-1.13.0", 1_000_000, 500_000)
	if !ok {
		t.Fatal("ComputeWithOK reported the model as unpriced")
	}
	if math.Abs(withOutput-got) > 1e-9 {
		t.Errorf("cost changed when output tokens were added (%v vs %v); output is free", withOutput, got)
	}
}

// The distinction HAL-531 exists to protect: a genuinely-zero rate is a
// real price, and must stay distinguishable from a model that has no entry
// at all. Both compute $0 for zero tokens — only the ok flag separates them.
func TestFreeOutputIsNotTheSameAsUnpriced(t *testing.T) {
	if _, ok := ComputeWithOK("jev-1.13.0", 0, 1000); !ok {
		t.Error("a free-output model reported as unpriced; the zero rate is real, not missing")
	}
	if _, ok := ComputeWithOK("definitely-not-a-model-xyz", 0, 1000); ok {
		t.Error("an unknown model reported as priced")
	}
}

// The hand-maintained table is the last layer, so anything an explicit
// Register knows still wins. Without this, a caller could never correct a
// stale System One rate without editing the library.
func TestRegisterOverridesTheSystemOneTable(t *testing.T) {
	const model = "jev-1.13.0"
	original, _ := Lookup(model)
	t.Cleanup(func() { Unregister(model) })

	Register(model, Price{InputPerMTok: 9.99, OutputPerMTok: 1.11})

	got, ok := Lookup(model)
	if !ok {
		t.Fatal("Lookup after Register = not found")
	}
	if got.InputPerMTok != 9.99 {
		t.Errorf("InputPerMTok = %v, want the registered 9.99", got.InputPerMTok)
	}

	Unregister(model)
	restored, ok := Lookup(model)
	if !ok || restored.InputPerMTok != original.InputPerMTok {
		t.Errorf("after Unregister, price = %+v, want the table's %+v", restored, original)
	}
}

// The System One table must not shadow anything in the generated defaults.
// A collision would mean a hand-maintained rate silently winning over a
// freshly generated one for some unrelated model.
func TestSystemOneTableDoesNotCollideWithDefaults(t *testing.T) {
	for id := range systemOnePrices {
		if _, clash := defaultPrices[id]; clash {
			t.Errorf("%q is in both systemOnePrices and the generated defaults; remove the hand-maintained entry", id)
		}
	}
}
