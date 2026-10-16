package pricing

import "testing"

// gemini-2.5-flash-lite used to resolve by prefix to gemini-2.5-flash and
// price at three times its input rate. The hand table must win.
func TestFlashLiteIsNotPricedAsFlash(t *testing.T) {
	p, ok := Lookup("gemini-2.5-flash-lite")
	if !ok {
		t.Fatal("gemini-2.5-flash-lite is unpriced")
	}
	if p.InputPerMTok != 0.10 || p.OutputPerMTok != 0.40 {
		t.Errorf("gemini-2.5-flash-lite = %+v, want $0.10/$0.40", p)
	}
	// The older Flash keeps its own rate.
	if p, _ := Lookup("gemini-2.5-flash"); p.InputPerMTok != 0.3 {
		t.Errorf("gemini-2.5-flash input = %v, want 0.3", p.InputPerMTok)
	}
}

func TestGemini3Prices(t *testing.T) {
	cases := map[string]Price{
		"gemini-3.5-flash-lite": {InputPerMTok: 0.30, OutputPerMTok: 2.50},
		"gemini-3.5-flash":      {InputPerMTok: 1.50, OutputPerMTok: 9.00},
		"gemini-3.8-flash":      {InputPerMTok: 1.50, OutputPerMTok: 7.50},
		// A versioned or preview suffix resolves by prefix.
		"gemini-3.1-pro-preview":    {InputPerMTok: 2.00, OutputPerMTok: 12.00},
		"gemini-3.5-flash-lite-001": {InputPerMTok: 0.30, OutputPerMTok: 2.50},
	}
	for id, want := range cases {
		p, ok := Lookup(id)
		if !ok {
			t.Errorf("%s is unpriced", id)
			continue
		}
		if p.InputPerMTok != want.InputPerMTok || p.OutputPerMTok != want.OutputPerMTok {
			t.Errorf("%s = $%v/$%v, want $%v/$%v", id, p.InputPerMTok, p.OutputPerMTok, want.InputPerMTok, want.OutputPerMTok)
		}
	}
}

func TestGoogleTableEntriesAreKnown(t *testing.T) {
	known := map[string]bool{}
	for _, m := range KnownModels() {
		known[m] = true
	}
	for id := range googlePrices {
		if !known[id] {
			t.Errorf("KnownModels omits %s", id)
		}
	}
}

// A remote snapshot that knows gemini-2.5-flash but not -lite must not
// price -lite by prefix as Flash.
func TestRemotePrefixDoesNotMaskGoogleTable(t *testing.T) {
	remote.Store(&snapshot{prices: map[string]Price{"gemini-2.5-flash": {InputPerMTok: 0.3, OutputPerMTok: 2.5}}})
	defer remote.Store(nil)
	p, ok := Lookup("gemini-2.5-flash-lite")
	if !ok || p.InputPerMTok != 0.10 {
		t.Errorf("gemini-2.5-flash-lite = %+v (%v), want the $0.10 hand-table rate", p, ok)
	}
	// An exact remote entry still wins over the hand table.
	remote.Store(&snapshot{prices: map[string]Price{"gemini-2.5-flash-lite": {InputPerMTok: 0.11, OutputPerMTok: 0.44}}})
	if p, _ := Lookup("gemini-2.5-flash-lite"); p.InputPerMTok != 0.11 {
		t.Errorf("exact remote entry should win, got %+v", p)
	}
}
