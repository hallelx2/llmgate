package llmgate

import (
	"errors"
	"testing"
)

// The primitives declare their methods on value receivers, which puts those
// methods in the method set of the pointer type too — so *Noul satisfies
// Question, a nil one passes an `x == nil` interface check, and validate()
// then dereferences it. That is a panic in a library, reachable from an
// ordinary caller slip.
func TestValidateRejectsTypedNilQuestions(t *testing.T) {
	var (
		nilNoul   *Noul
		nilChoice *Choice
		nilScore  *Score
	)

	cases := []struct {
		name string
		q    Question
	}{
		{"typed-nil *Noul", nilNoul},
		{"typed-nil *Choice", nilChoice},
		{"typed-nil *Score", nilScore},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("Validate() panicked on a typed-nil question: %v", r)
				}
			}()

			req := JudgeRequest{
				State:     "state",
				Questions: map[string]Question{"q": c.q},
			}
			if err := req.Validate(); !errors.Is(err, ErrQuestionInvalid) {
				t.Errorf("Validate() error = %v, want ErrQuestionInvalid", err)
			}
		})
	}
}

// A non-nil pointer to a valid question still has to work — the nil guard
// must not reject every pointer form.
func TestValidateAcceptsNonNilPointerQuestions(t *testing.T) {
	req := JudgeRequest{
		State:     "state",
		Questions: map[string]Question{"q": &Noul{Instructions: "urgent?"}},
	}
	if err := req.Validate(); err != nil {
		t.Errorf("Validate() = %v, want nil for a valid *Noul", err)
	}
}

// The mock must not let a later mutation rewrite what was recorded — an
// assertion that describes the wrong moment reads as a flake.
func TestMockJudgeRecordIsNotAliased(t *testing.T) {
	m := &MockJudge{}
	questions := map[string]Question{"a": Noul{Instructions: "a?"}}

	if _, err := m.Judge(t.Context(), JudgeRequest{State: "s", Questions: questions}); err != nil {
		t.Fatalf("Judge: %v", err)
	}

	// The caller reuses the map for a second, different call.
	questions["b"] = Noul{Instructions: "b?"}

	recorded, ok := m.LastRequest()
	if !ok {
		t.Fatal("LastRequest() = false after a call")
	}
	if len(recorded.Questions) != 1 {
		t.Errorf("recorded request now has %d questions; the caller's later mutation leaked into it", len(recorded.Questions))
	}
}
