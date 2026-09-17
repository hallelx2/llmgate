//go:build live

package typesafe_test

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/judge/typesafe"
)

// Live checks against the real API. Skipped unless a key is available, so
// `go test ./...` stays offline and deterministic by default.
//
// A stub server proves the wire handling. It cannot prove the two things
// that actually decide whether this seam is worth adopting: how fast a real
// batch comes back, and whether the model's answers are sane. Those need a
// real call, and a test is a better home for it than a shell one-liner
// because it can assert rather than just print.
//
// Run with either:
//
//	TYPESAFE_API_KEY=... go test -v -run TestLive ./judge/typesafe/
//	# or put the key in a gitignored .env at the repo root

func liveKey(t *testing.T) string {
	t.Helper()
	if k := os.Getenv(typesafe.EnvAPIKey); k != "" {
		return k
	}
	if k := keyFromDotEnv(); k != "" {
		return k
	}
	t.Skipf("no %s in the environment or a .env up-tree; skipping the live check", typesafe.EnvAPIKey)
	return ""
}

// keyFromDotEnv walks up from the test's directory looking for a .env with
// the key in it, so the credential never has to be pasted into a command.
func keyFromDotEnv() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for i := 0; i < 6; i++ {
		if v := readDotEnvValue(filepath.Join(dir, ".env"), typesafe.EnvAPIKey); v != "" {
			return v
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
}

func readDotEnvValue(path, key string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		name, value, ok := strings.Cut(line, "=")
		if !ok || strings.TrimSpace(name) != key {
			continue
		}
		return strings.Trim(strings.TrimSpace(value), `"'`)
	}
	return ""
}

const liveTicket = `Subject: payouts still failing

This is the third day my payouts have bounced. I've been charged the
$49 platform fee twice this month and nobody has answered my last two
emails. If this isn't fixed today I want the duplicate charge back and
I'm closing the account.`

// TestLiveBatch is the one that matters: four questions, one request,
// against the real model.
func TestLiveBatch(t *testing.T) {
	key := liveKey(t)

	j, err := typesafe.New(typesafe.Config{APIKey: key})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	start := time.Now()
	got, err := j.Judge(ctx, llmgate.JudgeRequest{
		Model: "jev-1.13.0",
		State: map[string]any{"ticket": liveTicket},
		Questions: map[string]llmgate.Question{
			"is_urgent": llmgate.Noul{
				Instructions: "Does the customer treat this as time-sensitive?",
				Criteria: &llmgate.NoulCriteria{
					True:  "States a deadline, or describes an ongoing loss",
					False: "No time pressure expressed",
				},
			},
			"wants_refund": llmgate.Noul{
				Instructions: "Is the customer asking for money back?",
			},
			"team": llmgate.Choice{
				Instructions: "Which team should own this ticket?",
				Options: llmgate.ChoiceOptions{
					{Name: "billing", Description: "Payments, invoicing, refunds, duplicate charges"},
					{Name: "technical", Description: "Bugs, outages, failed integrations"},
					{Name: "account", Description: "Cancellations, plan changes, access"},
					{Name: "unclear", Description: "The ticket does not say enough to route it"},
				},
			},
			"frustration": llmgate.Score{
				Instructions: "How frustrated does the customer sound?",
				Levels: []string{
					"Neutral — reporting a fact with no complaint",
					"Annoyed — impatient but still cooperative",
					"Angry — explicit complaint about being ignored or mistreated",
					"Leaving — threatening to cancel, refund, or escalate",
				},
			},
		},
	})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}

	t.Logf("model    %s", got.Model)
	t.Logf("elapsed  %s  (4 questions, 1 request)", elapsed.Round(time.Millisecond))
	t.Logf("tokens   %d in / %d out (reported=%v)",
		got.Usage.InputTokens, got.Usage.OutputTokens, got.Usage.TokensReported)
	t.Logf("cost     $%.8f (priced=%v)", got.Usage.CostUSD, got.Usage.Priced)

	// --- contract, not vibes ---

	if len(got.Answers) != 4 {
		t.Errorf("got %d answers, want 4", len(got.Answers))
	}
	if got.Model == "" {
		t.Error("response reported no model")
	}
	if !got.Usage.TokensReported {
		t.Error("TokensReported = false; the API documents a usage object")
	}
	if !got.Usage.Priced {
		t.Error("Priced = false; jev-1.13.0 is in the price book")
	}
	if got.Usage.CostUSD <= 0 {
		t.Errorf("CostUSD = %v, want positive", got.Usage.CostUSD)
	}

	urgent, err := got.Noul("is_urgent")
	if err != nil {
		t.Fatalf("is_urgent: %v", err)
	}
	refund, err := got.Noul("wants_refund")
	if err != nil {
		t.Fatalf("wants_refund: %v", err)
	}
	t.Logf("urgent        p=%.3f", urgent)
	t.Logf("wants refund  p=%.3f", refund)

	team, err := got.Choice("team")
	if err != nil {
		t.Fatalf("team: %v", err)
	}
	t.Logf("team          %s (confidence %.3f)", team.Choice, team.Confidence)
	for _, o := range sortedProbs(team.Probabilities) {
		t.Logf("                %-10s %.3f", o.name, o.p)
	}

	mood, err := got.Score("frustration")
	if err != nil {
		t.Fatalf("frustration: %v", err)
	}
	t.Logf("frustration   %.2f (confidence %.3f)", mood.Score, mood.Confidence)

	// Distributions must be well-formed, or every threshold built on them
	// is meaningless.
	if err := sumsToOne(team.Probabilities); err != nil {
		t.Errorf("choice distribution: %v", err)
	}
	intProbs := make(map[string]float64, len(mood.Probabilities))
	for i, p := range mood.Probabilities {
		intProbs[string(rune('0'+i))] = p
	}
	if err := sumsToOne(intProbs); err != nil {
		t.Errorf("score distribution: %v", err)
	}
	if len(mood.Legend) != 4 {
		t.Errorf("legend has %d entries, want one per level", len(mood.Legend))
	}
	if mood.Score < 0 || mood.Score > 3 {
		t.Errorf("score %v is outside the 0-3 level range", mood.Score)
	}

	// --- sanity, stated as a floor rather than an exact value ---
	//
	// The ticket explicitly demands a duplicate charge back, names a
	// deadline, and threatens to close the account. A model that reads
	// none of that is not usable for routing, whatever its latency. These
	// are deliberately loose: the point is to catch a wired-up-wrong
	// integration, not to pin the model's exact calibration.
	if refund <= 0.5 {
		t.Errorf("wants_refund = %.3f on a ticket that says \"I want the duplicate charge back\"", refund)
	}
	if urgent <= 0.5 {
		t.Errorf("is_urgent = %.3f on a ticket that says \"if this isn't fixed today\"", urgent)
	}
	if team.Choice != "billing" {
		t.Errorf("team = %q, want billing for a duplicate-charge refund", team.Choice)
	}
	if mood.Score < 1 {
		t.Errorf("frustration = %.2f; the customer is threatening to leave", mood.Score)
	}
}

// TestLiveRejectsABadKey confirms a 401 maps to the auth class rather than
// something retryable — worth checking against the real service, since the
// error body is the provider's, not ours.
func TestLiveRejectsABadKey(t *testing.T) {
	liveKey(t) // skip unless live testing was requested at all

	j, err := typesafe.New(typesafe.Config{APIKey: "apikey_definitely_not_valid"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	_, err = j.Judge(ctx, llmgate.JudgeRequest{
		State:     "anything",
		Questions: map[string]llmgate.Question{"q": llmgate.Noul{Instructions: "is this a test?"}},
	})
	if err == nil {
		t.Fatal("Judge() with a bad key = nil error")
	}
	if got := llmgate.Classify(err); got != llmgate.ErrClassAuth {
		t.Errorf("Classify = %v, want auth (got error: %v)", got, err)
	}
	if llmgate.IsTransient(err) {
		t.Error("a bad key is retryable; it does not fix itself")
	}
}

type prob struct {
	name string
	p    float64
}

func sortedProbs(m map[string]float64) []prob {
	out := make([]prob, 0, len(m))
	for n, p := range m {
		out = append(out, prob{n, p})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].p > out[j].p })
	return out
}

// sumsToOne checks a distribution is well-formed. A distribution that does
// not sum to 1 makes every threshold built on it meaningless.
func sumsToOne(m map[string]float64) error {
	if len(m) == 0 {
		return fmt.Errorf("distribution is empty")
	}
	var sum float64
	for _, p := range m {
		if p < 0 || p > 1 {
			return fmt.Errorf("probability %v is outside [0,1]", p)
		}
		sum += p
	}
	if sum < 0.99 || sum > 1.01 {
		return fmt.Errorf("probabilities sum to %v, want 1", sum)
	}
	return nil
}
