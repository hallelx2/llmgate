package llmgate

import (
	"context"
	"errors"
	"math"
	"testing"
)

func threeQuestions() JudgeRequest {
	return JudgeRequest{
		Model: "jev-1.13.0",
		State: "a support ticket",
		Questions: map[string]Question{
			"urgent": Noul{Instructions: "urgent?"},
			"team": Choice{Instructions: "which team?", Options: ChoiceOptions{
				{Name: "billing"}, {Name: "technical"}, {Name: "sales"},
			}},
			"frustration": Score{Instructions: "how frustrated?", Levels: []string{"calm", "annoyed", "angry"}},
		},
	}
}

func TestMockJudgeAnswersEveryQuestionInOneCall(t *testing.T) {
	m := &MockJudge{}

	got, err := m.Judge(context.Background(), threeQuestions())
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if m.Calls() != 1 {
		t.Errorf("Calls() = %d for a 3-question batch, want 1", m.Calls())
	}
	if len(got.Answers) != 3 {
		t.Errorf("got %d answers, want 3", len(got.Answers))
	}
}

// The default answers must be usable, not degenerate. An all-zero
// distribution would let a caller's thresholding pass without ever
// branching, and the test would prove nothing.
func TestMockJudgeDefaultDistributionsSumToOne(t *testing.T) {
	m := &MockJudge{}

	got, err := m.Judge(context.Background(), threeQuestions())
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}

	c, err := got.Choice("team")
	if err != nil {
		t.Fatalf("Choice: %v", err)
	}
	var choiceSum float64
	for _, p := range c.Probabilities {
		choiceSum += p
	}
	if math.Abs(choiceSum-1) > 1e-9 {
		t.Errorf("choice probabilities sum to %v, want 1", choiceSum)
	}
	if len(c.Probabilities) != 3 {
		t.Errorf("choice distribution has %d entries, want one per option", len(c.Probabilities))
	}
	if c.Choice == "" {
		t.Error("choice answer has no winner")
	}

	s, err := got.Score("frustration")
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	var scoreSum float64
	for _, p := range s.Probabilities {
		scoreSum += p
	}
	if math.Abs(scoreSum-1) > 1e-9 {
		t.Errorf("score probabilities sum to %v, want 1", scoreSum)
	}
	if len(s.Legend) != 3 || s.Legend[2] != "angry" {
		t.Errorf("legend = %v, want one entry per level keyed by index", s.Legend)
	}
	// The score has to be the probability-weighted position, matching how
	// the real model computes it.
	var want float64
	for i, p := range s.Probabilities {
		want += float64(i) * p
	}
	if math.Abs(s.Score-want) > 1e-9 {
		t.Errorf("Score = %v, want the probability-weighted %v", s.Score, want)
	}
}

func TestMockJudgeCannedAnswersWin(t *testing.T) {
	m := &MockJudge{Answers: map[string]Answer{
		"urgent": NoulAnswer{Noul: 0.99},
	}}

	got, err := m.Judge(context.Background(), threeQuestions())
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if n, err := got.Noul("urgent"); err != nil || n != 0.99 {
		t.Errorf("Noul = %v, %v; want the canned 0.99", n, err)
	}
	// The questions with no canned answer still get a default, so a test
	// only has to pin the values it actually cares about.
	if _, err := got.Choice("team"); err != nil {
		t.Errorf("uncanned question got no default answer: %v", err)
	}
}

func TestMockJudgeReturnsInjectedError(t *testing.T) {
	sentinel := errors.New("provider exploded")
	m := &MockJudge{Err: sentinel}

	if _, err := m.Judge(context.Background(), threeQuestions()); !errors.Is(err, sentinel) {
		t.Errorf("Judge() error = %v, want the injected sentinel", err)
	}
}

func TestMockJudgeSimulatesTheContextLimit(t *testing.T) {
	m := &MockJudge{}
	m.SimulateContextLimit()

	_, err := m.Judge(context.Background(), threeQuestions())
	if err == nil {
		t.Fatal("Judge() = nil error, want the simulated context-limit failure")
	}
	if got := Classify(err); got != ErrClassContextLength {
		t.Errorf("Classify = %v, want context_length", got)
	}
}

// Validating by default means a test cannot pass with a question the real
// transport would refuse to send.
func TestMockJudgeValidatesByDefault(t *testing.T) {
	m := &MockJudge{}

	_, err := m.Judge(context.Background(), JudgeRequest{
		State:     "s",
		Questions: map[string]Question{"bad": Score{Instructions: "rate", Levels: []string{"only"}}},
	})
	if !errors.Is(err, ErrQuestionInvalid) {
		t.Errorf("Judge() error = %v, want ErrQuestionInvalid", err)
	}

	m.SkipValidation = true
	if _, err := m.Judge(context.Background(), JudgeRequest{
		State:     "s",
		Questions: map[string]Question{"bad": Score{Instructions: "rate", Levels: []string{"only"}}},
	}); err != nil {
		t.Errorf("Judge() with SkipValidation: %v", err)
	}
}

func TestMockJudgeRecordsRequests(t *testing.T) {
	m := &MockJudge{}

	if _, ok := m.LastRequest(); ok {
		t.Error("LastRequest() reported a request before any call")
	}
	if _, err := m.Judge(context.Background(), threeQuestions()); err != nil {
		t.Fatalf("Judge: %v", err)
	}

	last, ok := m.LastRequest()
	if !ok {
		t.Fatal("LastRequest() = false after a call")
	}
	if len(last.Questions) != 3 || last.State != "a support ticket" {
		t.Errorf("recorded request = %+v, want the one that was sent", last)
	}
	if len(m.Requests()) != 1 {
		t.Errorf("Requests() has %d entries, want 1", len(m.Requests()))
	}
}

func TestMockJudgeReportsTheRequestedModel(t *testing.T) {
	m := &MockJudge{}
	got, err := m.Judge(context.Background(), threeQuestions())
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if got.Model != "jev-1.13.0" {
		t.Errorf("Model = %q, want the request's model", got.Model)
	}

	m2 := &MockJudge{Model: "jev-9.9.9"}
	got2, err := m2.Judge(context.Background(), threeQuestions())
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if got2.Model != "jev-9.9.9" {
		t.Errorf("Model = %q, want the mock's override", got2.Model)
	}
}
