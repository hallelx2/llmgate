package llmgate

import (
	"encoding/json"
	"errors"
	"testing"
)

// The expected payloads below are transcribed from the TypeSafe API
// reference (docs.typesafe.ai/api.md, read 2026-09-17). They are the
// contract: if a change here needs the expectation edited, the wire format
// changed and the transport needs looking at too.

func TestNoulMarshalsToDocumentedShape(t *testing.T) {
	q := Noul{
		Instructions: "Does this convey urgency?",
		Criteria: &NoulCriteria{
			True:  "Explicitly time-sensitive",
			False: "No urgency expressed",
		},
	}

	got := mustMarshal(t, q)
	want := `{"type":"noul","instructions":"Does this convey urgency?","criteria":{"true":"Explicitly time-sensitive","false":"No urgency expressed"}}`
	if got != want {
		t.Errorf("noul payload mismatch\n got: %s\nwant: %s", got, want)
	}
}

func TestNoulOmitsCriteriaWhenNil(t *testing.T) {
	got := mustMarshal(t, Noul{Instructions: "Does this convey urgency?"})
	want := `{"type":"noul","instructions":"Does this convey urgency?"}`
	if got != want {
		t.Errorf("noul payload mismatch\n got: %s\nwant: %s", got, want)
	}
}

func TestChoiceMarshalsToDocumentedShape(t *testing.T) {
	q := Choice{
		Instructions: "Which team should handle this?",
		Options: ChoiceOptions{
			{Name: "billing", Description: "Payments, invoicing, refunds"},
			{Name: "technical", Description: "Bugs, outages, integrations"},
			{Name: "sales", Description: "Pricing, upgrades, new accounts"},
		},
	}

	got := mustMarshal(t, q)
	want := `{"type":"choice","instructions":"Which team should handle this?",` +
		`"criteria":{"billing":"Payments, invoicing, refunds",` +
		`"technical":"Bugs, outages, integrations",` +
		`"sales":"Pricing, upgrades, new accounts"}}`
	if got != want {
		t.Errorf("choice payload mismatch\n got: %s\nwant: %s", got, want)
	}
}

// Declaration order has to survive marshalling. Go sorts map keys, so a
// map-backed implementation would silently reorder the options into
// alphabetical order and the payload would stop reading the way its author
// wrote it.
func TestChoicePreservesOptionOrder(t *testing.T) {
	q := Choice{
		Instructions: "pick",
		Options: ChoiceOptions{
			{Name: "zebra", Description: "last alphabetically, first declared"},
			{Name: "alpha", Description: "first alphabetically, last declared"},
		},
	}

	got := mustMarshal(t, q)
	want := `{"type":"choice","instructions":"pick",` +
		`"criteria":{"zebra":"last alphabetically, first declared",` +
		`"alpha":"first alphabetically, last declared"}}`
	if got != want {
		t.Errorf("option order not preserved\n got: %s\nwant: %s", got, want)
	}
}

// The API accepts null for an option that needs no rubric. An empty string
// is not the same thing on the wire.
func TestChoiceEmptyDescriptionMarshalsAsNull(t *testing.T) {
	q := Choice{
		Instructions: "pick",
		Options: ChoiceOptions{
			{Name: "yes"},
			{Name: "no", Description: "the negative case"},
		},
	}

	got := mustMarshal(t, q)
	want := `{"type":"choice","instructions":"pick","criteria":{"yes":null,"no":"the negative case"}}`
	if got != want {
		t.Errorf("null description mismatch\n got: %s\nwant: %s", got, want)
	}
}

func TestScoreMarshalsToDocumentedShape(t *testing.T) {
	q := Score{
		Instructions: "How frustrated is the customer?",
		Levels:       []string{"Calm", "Frustrated", "Very angry"},
	}

	got := mustMarshal(t, q)
	want := `{"type":"score","instructions":"How frustrated is the customer?","criteria":["Calm","Frustrated","Very angry"]}`
	if got != want {
		t.Errorf("score payload mismatch\n got: %s\nwant: %s", got, want)
	}
}

// Instructions accept structure, not only a string — the API allows an
// object or array when definitions or examples are needed to disambiguate.
func TestStructuredInstructionsSurvive(t *testing.T) {
	q := Noul{
		Instructions: map[string]any{
			"question": "Is this a refund request?",
			"excludes": []string{"chargeback", "cancellation"},
		},
	}

	got := mustMarshal(t, q)
	want := `{"type":"noul","instructions":{"excludes":["chargeback","cancellation"],"question":"Is this a refund request?"}}`
	if got != want {
		t.Errorf("structured instructions mismatch\n got: %s\nwant: %s", got, want)
	}
}

func TestQuestionKinds(t *testing.T) {
	cases := []struct {
		q    Question
		want QuestionKind
	}{
		{Noul{Instructions: "x"}, KindNoul},
		{Choice{Instructions: "x"}, KindChoice},
		{Score{Instructions: "x"}, KindScore},
	}
	for _, c := range cases {
		if got := c.q.Kind(); got != c.want {
			t.Errorf("Kind() = %q, want %q", got, c.want)
		}
	}
}

func TestValidateRejectsMalformedQuestions(t *testing.T) {
	cases := []struct {
		name string
		q    Question
	}{
		{"noul with no instructions", Noul{}},
		{"noul with empty instructions", Noul{Instructions: ""}},
		{"score with one level", Score{Instructions: "rate", Levels: []string{"only"}}},
		{"score with no levels", Score{Instructions: "rate"}},
		{"score with an empty level", Score{Instructions: "rate", Levels: []string{"low", ""}}},
		{"choice with one option", Choice{Instructions: "pick", Options: ChoiceOptions{{Name: "a"}}}},
		{"choice with no options", Choice{Instructions: "pick"}},
		{"choice with an unnamed option", Choice{Instructions: "pick", Options: ChoiceOptions{{Name: "a"}, {Name: ""}}}},
		{"choice with duplicate options", Choice{Instructions: "pick", Options: ChoiceOptions{{Name: "a"}, {Name: "a"}}}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req := JudgeRequest{Questions: map[string]Question{"q": c.q}}
			err := req.Validate()
			if err == nil {
				t.Fatalf("Validate() = nil, want an error")
			}
			if !errors.Is(err, ErrQuestionInvalid) {
				t.Errorf("Validate() error = %v, want it to wrap ErrQuestionInvalid", err)
			}
		})
	}
}

func TestValidateRejectsEmptyAndNilQuestions(t *testing.T) {
	t.Run("no questions", func(t *testing.T) {
		err := JudgeRequest{State: "anything"}.Validate()
		if !errors.Is(err, ErrNoQuestions) {
			t.Errorf("Validate() error = %v, want ErrNoQuestions", err)
		}
	})

	t.Run("nil question value", func(t *testing.T) {
		req := JudgeRequest{Questions: map[string]Question{"q": nil}}
		if err := req.Validate(); !errors.Is(err, ErrQuestionInvalid) {
			t.Errorf("Validate() error = %v, want ErrQuestionInvalid", err)
		}
	})
}

func TestValidateAcceptsAWellFormedBatch(t *testing.T) {
	req := JudgeRequest{
		Model: "jev-1.13.0",
		State: map[string]any{"message": "my payouts have been failing"},
		Questions: map[string]Question{
			"urgent": Noul{Instructions: "Does this convey urgency?"},
			"team": Choice{Instructions: "Which team?", Options: ChoiceOptions{
				{Name: "billing"}, {Name: "technical"},
			}},
			"frustration": Score{Instructions: "How frustrated?", Levels: []string{"calm", "angry"}},
		},
	}
	if err := req.Validate(); err != nil {
		t.Errorf("Validate() = %v, want nil", err)
	}
}

// A request with several bad questions must name the same one every run.
// Map iteration order is randomised, so an unsorted implementation would
// produce a flaky error message and an unreproducible bug report.
func TestValidateIsDeterministicAcrossRuns(t *testing.T) {
	req := JudgeRequest{Questions: map[string]Question{
		"a_bad": Score{Instructions: "rate", Levels: []string{"one"}},
		"b_bad": Choice{Instructions: "pick", Options: ChoiceOptions{{Name: "only"}}},
		"c_bad": Noul{},
	}}

	first := req.Validate()
	if first == nil {
		t.Fatal("Validate() = nil, want an error")
	}
	for i := 0; i < 50; i++ {
		if got := req.Validate(); got.Error() != first.Error() {
			t.Fatalf("Validate() is nondeterministic:\n first: %v\n  then: %v", first, got)
		}
	}
}

func TestJudgmentAccessorsReturnTypedAnswers(t *testing.T) {
	j := &Judgment{
		Model: "jev-1.13.0",
		Answers: map[string]Answer{
			"urgent": NoulAnswer{Noul: 0.92},
			"team": ChoiceAnswer{
				Choice:        "technical",
				Probabilities: map[string]float64{"billing": 0.08, "technical": 0.85, "sales": 0.07},
				Confidence:    0.82,
			},
			"frustration": ScoreAnswer{
				Score:         1.6,
				Legend:        map[int]string{0: "Calm", 1: "Frustrated", 2: "Very angry"},
				Probabilities: map[int]float64{0: 0.05, 1: 0.3, 2: 0.65},
				Confidence:    0.78,
			},
		},
	}

	if got, err := j.Noul("urgent"); err != nil || got != 0.92 {
		t.Errorf("Noul() = %v, %v; want 0.92, nil", got, err)
	}

	c, err := j.Choice("team")
	if err != nil {
		t.Fatalf("Choice() error = %v", err)
	}
	if c.Choice != "technical" || c.Confidence != 0.82 || len(c.Probabilities) != 3 {
		t.Errorf("Choice() = %+v, want technical with 3 probabilities and confidence 0.82", c)
	}

	s, err := j.Score("frustration")
	if err != nil {
		t.Fatalf("Score() error = %v", err)
	}
	if s.Score != 1.6 || s.Legend[2] != "Very angry" || s.Probabilities[2] != 0.65 {
		t.Errorf("Score() = %+v, want 1.6 with a populated legend and distribution", s)
	}
}

// Reading the wrong primitive has to be an error, not a zero value. A
// silent 0 from a mistyped accessor would flow straight into a threshold
// and read as "definitely no".
func TestJudgmentAccessorsRejectAMismatchedPrimitive(t *testing.T) {
	j := &Judgment{Answers: map[string]Answer{
		"team": ChoiceAnswer{Choice: "billing"},
	}}

	if _, err := j.Noul("team"); !errors.Is(err, ErrAnswerType) {
		t.Errorf("Noul() on a choice: error = %v, want ErrAnswerType", err)
	}
	if _, err := j.Score("team"); !errors.Is(err, ErrAnswerType) {
		t.Errorf("Score() on a choice: error = %v, want ErrAnswerType", err)
	}
}

func TestJudgmentAccessorsReportMissingAnswers(t *testing.T) {
	j := &Judgment{Answers: map[string]Answer{"present": NoulAnswer{Noul: 1}}}

	if _, err := j.Noul("absent"); !errors.Is(err, ErrAnswerMissing) {
		t.Errorf("Noul() on a missing id: error = %v, want ErrAnswerMissing", err)
	}
	if _, err := j.Choice("absent"); !errors.Is(err, ErrAnswerMissing) {
		t.Errorf("Choice() on a missing id: error = %v, want ErrAnswerMissing", err)
	}
}

// A caller that ignored the error from Judge holds a nil *Judgment. The
// accessors must report that rather than panicking.
func TestJudgmentAccessorsSurviveANilReceiver(t *testing.T) {
	var j *Judgment
	if _, err := j.Noul("anything"); !errors.Is(err, ErrAnswerMissing) {
		t.Errorf("Noul() on nil judgment: error = %v, want ErrAnswerMissing", err)
	}
}

func TestAnswerKinds(t *testing.T) {
	cases := []struct {
		a    Answer
		want QuestionKind
	}{
		{NoulAnswer{}, KindNoul},
		{ChoiceAnswer{}, KindChoice},
		{ScoreAnswer{}, KindScore},
	}
	for _, c := range cases {
		if got := c.a.Kind(); got != c.want {
			t.Errorf("Kind() = %q, want %q", got, c.want)
		}
	}
}

func TestLevelIndexParsesWireKeys(t *testing.T) {
	if got, err := levelIndex("2"); err != nil || got != 2 {
		t.Errorf("levelIndex(\"2\") = %d, %v; want 2, nil", got, err)
	}
	if _, err := levelIndex("two"); err == nil {
		t.Error("levelIndex(\"two\") = nil error, want a parse failure")
	}
}

func mustMarshal(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	return string(b)
}
