// Command judge is a one-shot live check against TypeSafe's System One API.
// Run with: TYPESAFE_API_KEY=... go run ./examples/judge
//
// It asks four questions about one support ticket in a single request:
// whether the ticket is urgent, whether it asks for a refund, which team
// should own it, and how frustrated the customer sounds.
//
// The point of the example is the shape, not the answers. Four judgments
// over one state cost one round-trip, because the model reads the state
// once and runs every question against it in parallel. The instinct to fan
// these out into four calls is the thing to unlearn.
package main

import (
	"context"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/judge/typesafe"
	"github.com/hallelx2/llmgate/middleware/retry"
)

const ticket = `Subject: payouts still failing

This is the third day my payouts have bounced. I've been charged the
$49 platform fee twice this month and nobody has answered my last two
emails. If this isn't fixed today I want the duplicate charge back and
I'm closing the account.`

func main() {
	j, err := typesafe.New(typesafe.Config{})
	if err != nil {
		fmt.Fprintln(os.Stderr, "construct:", err)
		os.Exit(1)
	}

	// Retry is worth wrapping even for a one-shot: a batched call loses
	// every answer in it when the request drops, and TypeSafe documents its
	// rate limits as changing without notice.
	judge := retry.NewJudge(retry.Config{MaxRetries: 3})(j)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	start := time.Now()
	result, err := judge.Judge(ctx, llmgate.JudgeRequest{
		// Pinned rather than an alias: an alias resolves server-side and
		// can start answering from a different model without warning.
		Model: "jev-1.13.0",
		State: map[string]any{"ticket": ticket},

		Questions: map[string]llmgate.Question{
			"is_urgent": llmgate.Noul{
				Instructions: "Does the customer treat this as time-sensitive?",
				Criteria: &llmgate.NoulCriteria{
					True:  "States a deadline, or describes an ongoing loss",
					False: "No time pressure expressed",
				},
			},

			// A second Noul rather than a Choice, because a ticket can be
			// both urgent and a refund request. A Choice would force one
			// winner and lose the other.
			"wants_refund": llmgate.Noul{
				Instructions: "Is the customer asking for money back?",
			},

			"team": llmgate.Choice{
				Instructions: "Which team should own this ticket?",
				Options: llmgate.ChoiceOptions{
					{Name: "billing", Description: "Payments, invoicing, refunds, duplicate charges"},
					{Name: "technical", Description: "Bugs, outages, failed integrations"},
					{Name: "account", Description: "Cancellations, plan changes, access"},
					// The no-match option. Without one the distribution is
					// forced across options that may all be wrong, and the
					// confidence describes a choice between bad answers
					// rather than the absence of a good one.
					{Name: "unclear", Description: "The ticket does not say enough to route it"},
				},
			},

			"frustration": llmgate.Score{
				Instructions: "How frustrated does the customer sound?",
				// Levels describe concrete situations. A bare ordinal gives
				// the model nothing to match against.
				Levels: []string{
					"Neutral — reporting a fact with no complaint",
					"Annoyed — the tone is impatient but still cooperative",
					"Angry — explicit complaint about being ignored or mistreated",
					"Leaving — threatening to cancel, refund, or escalate",
				},
			},
		},
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "judge:", err)
		os.Exit(1)
	}

	fmt.Printf("model    %s\n", result.Model)
	fmt.Printf("elapsed  %s\n", time.Since(start).Round(time.Millisecond))
	fmt.Printf("tokens   %d in / %d out\n", result.Usage.InputTokens, result.Usage.OutputTokens)
	if result.Usage.Priced {
		fmt.Printf("cost     $%.8f\n", result.Usage.CostUSD)
	} else {
		fmt.Printf("cost     unknown (no price-book entry)\n")
	}
	fmt.Println()

	urgent, err := result.Noul("is_urgent")
	if err != nil {
		fmt.Fprintln(os.Stderr, "is_urgent:", err)
		os.Exit(1)
	}
	refund, err := result.Noul("wants_refund")
	if err != nil {
		fmt.Fprintln(os.Stderr, "wants_refund:", err)
		os.Exit(1)
	}
	fmt.Printf("urgent        p=%.3f\n", urgent)
	fmt.Printf("wants refund  p=%.3f\n", refund)

	team, err := result.Choice("team")
	if err != nil {
		fmt.Fprintln(os.Stderr, "team:", err)
		os.Exit(1)
	}
	fmt.Printf("team          %s (confidence %.3f)\n", team.Choice, team.Confidence)
	for _, opt := range sortedByProbability(team.Probabilities) {
		fmt.Printf("                %-10s %.3f\n", opt.name, opt.p)
	}

	frustration, err := result.Score("frustration")
	if err != nil {
		fmt.Fprintln(os.Stderr, "frustration:", err)
		os.Exit(1)
	}
	fmt.Printf("frustration   %.2f (confidence %.3f)\n", frustration.Score, frustration.Confidence)
	for i := 0; i < len(frustration.Legend); i++ {
		fmt.Printf("                %d %-58s %.3f\n", i, frustration.Legend[i], frustration.Probabilities[i])
	}
	fmt.Println()

	// The decision belongs to code. The model supplied the judgments; the
	// policy that combines them — including the "any serious signal" rule,
	// which a weighted average would wash out — is ours to write and ours
	// to change without re-running inference.
	switch {
	case team.Confidence < 0.5:
		fmt.Println("route: human triage — no clear owner")
	case refund > 0.7 && frustration.Score >= 2:
		fmt.Println("route: billing, escalated — refund request from an angry customer")
	case urgent > 0.7:
		fmt.Printf("route: %s, priority queue\n", team.Choice)
	default:
		fmt.Printf("route: %s, normal queue\n", team.Choice)
	}
}

type scored struct {
	name string
	p    float64
}

// sortedByProbability orders a distribution highest-first so the output is
// readable and stable across runs.
func sortedByProbability(probs map[string]float64) []scored {
	out := make([]scored, 0, len(probs))
	for name, p := range probs {
		out = append(out, scored{name, p})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].p != out[j].p {
			return out[i].p > out[j].p
		}
		return out[i].name < out[j].name
	})
	return out
}
