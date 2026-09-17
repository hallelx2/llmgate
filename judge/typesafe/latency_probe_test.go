//go:build live

package typesafe_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/judge/typesafe"
)

// The question this probe exists to answer, for vectorless HAL-1352:
//
// The engine's TOC detector asks the same yes/no question of up to 20 pages
// in a SEQUENTIAL loop — one generative call per page. Replacing it with a
// Judge is only worth doing if a batch of N questions costs roughly what
// one costs, rather than N times as much.
//
// So: hold the state fixed, vary the question count, and see whether the
// curve is flat. A flat curve means batching is the win. A linear one means
// the model is answering serially behind the API and the whole premise is
// wrong.
//
// Run with:
//
//	go test -v -count=1 -run TestLiveLatencyScaling ./judge/typesafe/
func TestLiveLatencyScaling(t *testing.T) {
	key := liveKey(t)
	if testing.Short() {
		t.Skip("skipping the latency probe in -short mode; it makes several live calls")
	}

	j, err := typesafe.New(typesafe.Config{APIKey: key})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// One page of plausible document text, held constant across every run
	// so the only thing changing is the question count.
	const page = `TABLE OF CONTENTS

Item 1.  Business .............................................. 4
Item 1A. Risk Factors ......................................... 12
Item 2.  Properties ........................................... 28
Item 3.  Legal Proceedings .................................... 31
Item 5.  Market for Registrant's Common Equity ................ 33
Item 7.  Management's Discussion and Analysis ................. 36
Item 8.  Financial Statements and Supplementary Data .......... 52`

	counts := []int{1, 5, 10, 20}
	type row struct {
		n        int
		elapsed  time.Duration
		inTokens int
		cost     float64
	}
	var rows []row

	for _, n := range counts {
		questions := make(map[string]llmgate.Question, n)
		for i := 0; i < n; i++ {
			questions[fmt.Sprintf("page_%02d", i)] = llmgate.Noul{
				Instructions: "Does this text contain a table of contents?",
				Criteria: &llmgate.NoulCriteria{
					True:  "A list of sections or items with page numbers",
					False: "Ordinary body text, an abstract, or a list of figures or tables",
				},
			}
		}

		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		start := time.Now()
		got, err := j.Judge(ctx, llmgate.JudgeRequest{
			Model:     "jev-1.13.0",
			State:     map[string]any{"page_text": page},
			Questions: questions,
		})
		elapsed := time.Since(start)
		cancel()

		if err != nil {
			t.Fatalf("%d questions: %v", n, err)
		}
		if len(got.Answers) != n {
			t.Errorf("%d questions produced %d answers", n, len(got.Answers))
		}
		rows = append(rows, row{n, elapsed, got.Usage.InputTokens, got.Usage.CostUSD})
	}

	t.Log("questions | elapsed | input tokens | cost")
	for _, r := range rows {
		t.Logf("%9d | %7s | %12d | $%.8f",
			r.n, r.elapsed.Round(time.Millisecond), r.inTokens, r.cost)
	}

	// The claim under test. Twenty questions must not cost twenty times one
	// question's wall-clock — if it does, batching buys nothing and the
	// engine work should not proceed on a latency argument.
	one, twenty := rows[0].elapsed, rows[len(rows)-1].elapsed
	ratio := float64(twenty) / float64(one)
	t.Logf("20-question call is %.2fx the wall-clock of a 1-question call", ratio)
	if ratio > 5 {
		t.Errorf("latency scales close to linearly with question count (%.2fx for 20x the questions); "+
			"batching is not buying parallelism", ratio)
	}
}

// TestLiveSingleQuestionFloor measures the per-call floor: the smallest
// possible request, repeated, to separate fixed round-trip overhead from
// per-question work. This is the number that decides whether a per-item
// loop can ever be fast, no matter how small the item.
func TestLiveSingleQuestionFloor(t *testing.T) {
	key := liveKey(t)
	if testing.Short() {
		t.Skip("skipping the latency floor probe in -short mode")
	}

	j, err := typesafe.New(typesafe.Config{APIKey: key})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const runs = 5
	var total time.Duration
	var fastest, slowest time.Duration

	for i := 0; i < runs; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		start := time.Now()
		_, err := j.Judge(ctx, llmgate.JudgeRequest{
			Model:     "jev-1.13.0",
			State:     "The quick brown fox jumps over the lazy dog.",
			Questions: map[string]llmgate.Question{"q": llmgate.Noul{Instructions: "Does this mention an animal?"}},
		})
		elapsed := time.Since(start)
		cancel()
		if err != nil {
			t.Fatalf("run %d: %v", i, err)
		}

		total += elapsed
		if fastest == 0 || elapsed < fastest {
			fastest = elapsed
		}
		if elapsed > slowest {
			slowest = elapsed
		}
	}

	t.Logf("minimal 1-question call over %d runs: mean %s, fastest %s, slowest %s",
		runs, (total / runs).Round(time.Millisecond),
		fastest.Round(time.Millisecond), slowest.Round(time.Millisecond))
}
