package typesafe

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

// The chunked count must equal the serial count exactly, not
// approximately: the guard compares it against hard limits.
func TestChunkedCountEqualsSerialCount(t *testing.T) {
	enc, err := cl100k()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1708))
	words := []string{
		"Revenue", "was", "$12,345", "million,", "up", "4.5%", "(2021:", "11,812)", "—", "Note", "21.",
		"the", "Company's", "don't", "it'll", "\n\n", "\t", "  ", "café", "naïve", "東京", "£1.2bn", "10-K",
		"\"quoted\"", "{json}", "a", "I", "x2", "2022", "1234567", "...", "\r\n", "   indent", "émigré",
	}
	var texts []string
	for n := 0; n < 60; n++ {
		var b strings.Builder
		for b.Len() < 20_000+rng.Intn(60_000) {
			b.WriteString(words[rng.Intn(len(words))])
			if rng.Intn(5) > 0 {
				b.WriteByte(' ')
			}
		}
		texts = append(texts, b.String())
	}
	// And the shape the guard actually counts: a marshalled request state.
	state := map[string]any{"question": "What was FY2022 revenue?"}
	for i := 0; i < 16; i++ {
		state[fmt.Sprintf("p_%d", i)] = map[string]any{"page": i, "text": texts[i][:4500]}
	}
	sj, _ := json.Marshal(state)
	texts = append(texts, string(sj))

	for i, text := range texts {
		chunks := splitForCount(text, parallelCountBytes)
		if strings.Join(chunks, "") != text {
			t.Fatalf("text %d: chunks do not reassemble the text", i)
		}
		want := len(enc.Encode(text, nil, nil))
		got, err := EstimateTokens(text)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("text %d (%d bytes, %d chunks): chunked count %d, serial %d", i, len(text), len(chunks), got, want)
		}
	}
}

func TestSplitForCountLeavesShortTextWhole(t *testing.T) {
	if got := splitForCount("short text", parallelCountBytes); len(got) != 1 {
		t.Errorf("got %d chunks", len(got))
	}
	dense := strings.Repeat("9", 3*parallelCountBytes)
	if got := splitForCount(dense, parallelCountBytes); len(got) != 1 || got[0] != dense {
		t.Errorf("text with no safe cut must come back whole, got %d chunks", len(got))
	}
}
