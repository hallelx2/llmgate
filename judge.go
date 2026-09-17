package llmgate

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
)

// Judge is the contract for a System One model: one that reads application
// state and returns typed judgments rather than generated text.
//
// It is deliberately NOT a Client. Client is a chat-completion surface —
// roles, messages, an assistant turn carrying prose. A System One model has
// none of those. It takes a state and a map of named questions and answers
// every one of them, in parallel, with a value whose type the caller fixed
// in advance. Expressing that through Complete would mean encoding the
// questions into a prompt string and parsing answers back out of text,
// which reintroduces exactly the JSON-mode retry machinery the typed API
// exists to remove.
//
// So the two interfaces sit side by side. Hold a Client for work that must
// produce language; hold a Judge for work that must produce a decision. A
// caller that needs both holds both, and a caller that needs neither one of
// them holds nil — every consumer in this codebase treats a nil Judge as
// "judging disabled" and falls back to its generative path.
//
// The single most important property of this interface is that a batch of
// questions over one state is ONE call. Questions in the same request
// cannot see one another's answers, which is the constraint that makes them
// parallelisable; ask a second round only when an answer determines what
// evidence to fetch next.
type Judge interface {
	// Judge evaluates req.State against every question in req.Questions and
	// returns one answer per question, keyed by the same IDs.
	Judge(ctx context.Context, req JudgeRequest) (*Judgment, error)
}

// JudgeMiddleware wraps a Judge, mirroring Middleware for Client so the
// same retry / cache / budget composition style works on both seams.
type JudgeMiddleware func(Judge) Judge

// ProviderTypeSafe identifies TypeSafe, whose Jev model is the first
// System One implementation behind Judge.
const ProviderTypeSafe Provider = "typesafe"

// Sentinel errors for the judgment seam. All are wrapped with context by
// the callers that return them; branch with errors.Is.
var (
	// ErrNoQuestions is returned when a JudgeRequest carries no questions.
	// Sending it would bill for the state and return nothing.
	ErrNoQuestions = errors.New("llmgate: judge request has no questions")

	// ErrQuestionInvalid is returned when a question cannot be sent as
	// written — a Score with fewer than two levels, a Choice with no
	// options, missing instructions. Caught locally: the provider would
	// answer these with a 422, and a round-trip to learn that is waste.
	ErrQuestionInvalid = errors.New("llmgate: invalid question")

	// ErrAnswerType is returned by the Judgment accessors when a question
	// ID exists but was answered with a different primitive than the one
	// being read — asking for a Noul where a Choice was requested.
	ErrAnswerType = errors.New("llmgate: answer is a different primitive")

	// ErrAnswerMissing is returned by the Judgment accessors when no answer
	// came back under the requested ID.
	ErrAnswerMissing = errors.New("llmgate: no answer for question id")
)

// QuestionKind discriminates the three System One primitives. The values
// match the wire format's `type` field exactly.
type QuestionKind string

const (
	// KindNoul is a yes/no question answered with the probability of yes.
	KindNoul QuestionKind = "noul"
	// KindChoice selects one option from a defined set.
	KindChoice QuestionKind = "choice"
	// KindScore rates the state against ordered, described levels.
	KindScore QuestionKind = "score"
)

// Question is one typed question in a JudgeRequest. The interface is sealed
// — only Noul, Choice, and Score implement it — because the set of
// primitives is fixed by the model, not extensible by callers.
type Question interface {
	// Kind reports which primitive this question is.
	Kind() QuestionKind

	// validate reports whether the question can be sent as written.
	validate() error

	// sealed prevents implementations outside this package.
	sealed()
}

// Noul asks a yes/no question and is answered with the probability that
// the answer is yes.
//
// Use one Noul per label when several labels may apply at once; a Choice
// forces exactly one winner, which is wrong when two can be true together.
//
// A Noul answer carries no confidence value, and that is not an omission:
// the probability already is the answer. A Noul near 0.5 means the model
// finds yes and no roughly equally likely — it does NOT mean "medium
// intensity", and reading it that way is the most common way to misuse
// this primitive.
type Noul struct {
	// Instructions is the yes/no question to evaluate. A string is the
	// common case. An object or array is accepted for questions that need
	// definitions, contrasts, exclusions, or examples to be unambiguous.
	Instructions any

	// Criteria optionally describes what a yes and a no each mean. Leaving
	// it nil is fine when the question is self-evident.
	//
	// Keep it aligned with Instructions. A Noul whose True describes the
	// "no" case measurably degrades the answer — the model reads the
	// criteria as an extension of the question, not as a correction to it.
	Criteria *NoulCriteria
}

// NoulCriteria describes the two poles of a Noul.
type NoulCriteria struct {
	// True is what an answer near 1 means.
	True string `json:"true,omitempty"`
	// False is what an answer near 0 means.
	False string `json:"false,omitempty"`
}

// Kind implements Question.
func (Noul) Kind() QuestionKind { return KindNoul }

func (Noul) sealed() {}

func (n Noul) validate() error {
	return validateInstructions(n.Instructions)
}

// MarshalJSON renders the wire shape for a Noul question.
func (n Noul) MarshalJSON() ([]byte, error) {
	payload := struct {
		Type         QuestionKind  `json:"type"`
		Instructions any           `json:"instructions"`
		Criteria     *NoulCriteria `json:"criteria,omitempty"`
	}{KindNoul, n.Instructions, n.Criteria}
	return json.Marshal(payload)
}

// Choice selects exactly one option from a set the caller defines, and
// returns the full probability distribution across those options.
//
// Include a no-match option whenever none of the real options might fit.
// Without one the distribution is forced across options that are all wrong,
// and the confidence that comes back describes a choice between bad
// answers rather than the absence of a good one.
type Choice struct {
	// Instructions is what the model should decide.
	Instructions any

	// Options are the selectable outcomes, in the order the caller wrote
	// them. Order is preserved on the wire rather than sorted, so the
	// payload reads the way its author meant it.
	Options ChoiceOptions
}

// ChoiceOptions is an ordered set of selectable options. It marshals to a
// JSON object of option name to description, preserving declaration order.
type ChoiceOptions []ChoiceOption

// ChoiceOption is one selectable outcome and its rubric.
type ChoiceOption struct {
	// Name is the option the model returns when it picks this one. It is
	// also the key in the answer's probability distribution.
	Name string

	// Description is the rubric for this option. An empty Description
	// marshals as null, which the API accepts for an option that needs no
	// elaboration.
	Description string
}

// MarshalJSON renders the options as a JSON object in declaration order.
func (o ChoiceOptions) MarshalJSON() ([]byte, error) {
	if len(o) == 0 {
		return []byte("{}"), nil
	}
	buf := []byte{'{'}
	for i, opt := range o {
		if i > 0 {
			buf = append(buf, ',')
		}
		key, err := json.Marshal(opt.Name)
		if err != nil {
			return nil, fmt.Errorf("llmgate: marshal choice option %q: %w", opt.Name, err)
		}
		buf = append(buf, key...)
		buf = append(buf, ':')
		if opt.Description == "" {
			buf = append(buf, []byte("null")...)
			continue
		}
		desc, err := json.Marshal(opt.Description)
		if err != nil {
			return nil, fmt.Errorf("llmgate: marshal choice description for %q: %w", opt.Name, err)
		}
		buf = append(buf, desc...)
	}
	return append(buf, '}'), nil
}

// Kind implements Question.
func (Choice) Kind() QuestionKind { return KindChoice }

func (Choice) sealed() {}

func (c Choice) validate() error {
	if err := validateInstructions(c.Instructions); err != nil {
		return err
	}
	if len(c.Options) < 2 {
		return fmt.Errorf("%w: choice needs at least 2 options, got %d", ErrQuestionInvalid, len(c.Options))
	}
	seen := make(map[string]struct{}, len(c.Options))
	for _, opt := range c.Options {
		if opt.Name == "" {
			return fmt.Errorf("%w: choice option name is empty", ErrQuestionInvalid)
		}
		if _, dup := seen[opt.Name]; dup {
			// A duplicate key would silently collapse in the JSON object
			// and the distribution would come back with fewer entries than
			// the caller declared.
			return fmt.Errorf("%w: duplicate choice option %q", ErrQuestionInvalid, opt.Name)
		}
		seen[opt.Name] = struct{}{}
	}
	return nil
}

// MarshalJSON renders the wire shape for a Choice question.
func (c Choice) MarshalJSON() ([]byte, error) {
	payload := struct {
		Type         QuestionKind  `json:"type"`
		Instructions any           `json:"instructions"`
		Criteria     ChoiceOptions `json:"criteria"`
	}{KindChoice, c.Instructions, c.Options}
	return json.Marshal(payload)
}

// Score rates the state along an ordered rubric and returns a
// probability-weighted position across the levels, which can land between
// them.
//
// Each level must describe a concrete situation and stand on its own; a
// bare ordinal ("2") gives the model nothing to match against.
//
// The returned value is safe to threshold. It is NOT safe to treat as a
// measurement: interpolating between two levels to recover an exact
// magnitude is explicitly outside what the model is calibrated for.
type Score struct {
	// Instructions is what the model should rate.
	Instructions any

	// Levels are the ordered rubric levels, lowest first. At least two are
	// required. The answer's Legend maps each level index back to its
	// description here.
	Levels []string
}

// Kind implements Question.
func (Score) Kind() QuestionKind { return KindScore }

func (Score) sealed() {}

func (s Score) validate() error {
	if err := validateInstructions(s.Instructions); err != nil {
		return err
	}
	if len(s.Levels) < 2 {
		return fmt.Errorf("%w: score needs at least 2 levels, got %d", ErrQuestionInvalid, len(s.Levels))
	}
	for i, lvl := range s.Levels {
		if lvl == "" {
			return fmt.Errorf("%w: score level %d has an empty description", ErrQuestionInvalid, i)
		}
	}
	return nil
}

// MarshalJSON renders the wire shape for a Score question.
func (s Score) MarshalJSON() ([]byte, error) {
	payload := struct {
		Type         QuestionKind `json:"type"`
		Instructions any          `json:"instructions"`
		Criteria     []string     `json:"criteria"`
	}{KindScore, s.Instructions, s.Levels}
	return json.Marshal(payload)
}

// validateInstructions enforces that a question actually asks something.
// A nil or empty-string instruction is always a caller bug: the question ID
// is not sent to the model, so an empty instruction leaves it with nothing
// to answer.
func validateInstructions(instructions any) error {
	switch v := instructions.(type) {
	case nil:
		return fmt.Errorf("%w: instructions are empty", ErrQuestionInvalid)
	case string:
		if v == "" {
			return fmt.Errorf("%w: instructions are empty", ErrQuestionInvalid)
		}
	}
	return nil
}

// JudgeRequest is one evaluation: a state, and every question to ask about
// it.
type JudgeRequest struct {
	// Model selects the System One model. Empty means the implementation's
	// default.
	//
	// Prefer a pinned version over a moving alias once thresholds are tuned
	// against it: an alias resolves server-side and can start answering
	// from a different model without any change here.
	Model string

	// State is the content to evaluate — a string for plain text, or a
	// struct / map / slice for structured input.
	//
	// Send only what the questions need. Accuracy falls as the state grows
	// with material unrelated to the decision, and a large state also makes
	// it much harder to work out which part produced a wrong answer.
	State any

	// Questions is the batch to answer, keyed by IDs the caller chooses.
	// Answers come back under the same keys.
	//
	// The IDs are for the caller's code and are NOT sent to the model, so
	// the full meaning of a question has to live in its Instructions — a
	// descriptive key is not a substitute.
	Questions map[string]Question
}

// Validate reports whether the request can be sent as written. It checks
// structure only; size limits belong to the implementation, which knows the
// model's context budget.
func (r JudgeRequest) Validate() error {
	if len(r.Questions) == 0 {
		return ErrNoQuestions
	}
	// Sorted so a request with several bad questions reports the same one
	// on every run; map iteration order would make the error nondeterministic.
	for _, id := range sortedKeys(r.Questions) {
		q := r.Questions[id]
		if isNilQuestion(q) {
			return fmt.Errorf("%w: question %q is nil", ErrQuestionInvalid, id)
		}
		if err := q.validate(); err != nil {
			return fmt.Errorf("question %q: %w", id, err)
		}
	}
	return nil
}

// isNilQuestion reports whether q carries no usable value.
//
// A plain q == nil misses the typed-nil case. The primitives declare their
// methods on value receivers, which puts those methods in the method set of
// the pointer type too, so a *Noul satisfies Question — and a nil one
// passes an interface nil check, then panics when validate() dereferences
// it. The interface is sealed, so the three pointer forms below are the
// complete set that can reach here.
func isNilQuestion(q Question) bool {
	switch v := q.(type) {
	case nil:
		return true
	case *Noul:
		return v == nil
	case *Choice:
		return v == nil
	case *Score:
		return v == nil
	default:
		return false
	}
}

// Judgment is the result of one Judge call.
type Judgment struct {
	// Model is the model that actually answered, as the provider reported
	// it. When the request named an alias this is the versioned ID behind
	// it, which is the value worth logging: a threshold tuned against one
	// version is not evidence about the next.
	Model string

	// Answers holds one answer per question, under the ID the caller used.
	Answers map[string]Answer

	// Usage is the normalized token and cost accounting for the call.
	Usage Usage
}

// Answer is one typed answer. Type-switch on it, or use the Judgment
// accessors, which report a clear error instead of panicking on a
// mismatch.
type Answer interface {
	// Kind reports which primitive produced this answer.
	Kind() QuestionKind

	// sealed prevents implementations outside this package.
	sealed()
}

// NoulAnswer is the probability that the answer to a Noul is yes.
//
// It carries no confidence value. The probability is the whole answer, and
// there is no second axis to consult.
type NoulAnswer struct {
	// Noul is the probability of yes, in [0, 1].
	Noul float64
}

// Kind implements Answer.
func (NoulAnswer) Kind() QuestionKind { return KindNoul }

func (NoulAnswer) sealed() {}

// ChoiceAnswer is the selected option plus the full distribution.
type ChoiceAnswer struct {
	// Choice is the highest-probability option.
	Choice string

	// Probabilities maps every declared option to its probability. The
	// values sum to 1.
	Probabilities map[string]float64

	// Confidence in [0, 1] summarises how concentrated Probabilities is.
	//
	// It describes the shape of this one distribution and nothing else: it
	// is not a probability that the answer is correct, and it is not
	// permission to act. Low confidence on a choice between several
	// acceptable options is not a problem to route around.
	Confidence float64
}

// Kind implements Answer.
func (ChoiceAnswer) Kind() QuestionKind { return KindChoice }

func (ChoiceAnswer) sealed() {}

// ScoreAnswer is the probability-weighted position across a Score's levels.
type ScoreAnswer struct {
	// Score is the probability-weighted value across the levels. It can
	// land between two levels — 1.6 is a real answer, not a rounding
	// artefact.
	//
	// Threshold it. Do not interpolate between adjacent levels to recover
	// an exact magnitude; the levels are not calibrated as a numeric scale.
	Score float64

	// Legend maps each level index to the description the caller supplied,
	// so a caller can render the level the score sits near without keeping
	// the original Levels slice alive.
	Legend map[int]string

	// Probabilities maps each level index to its probability. The values
	// sum to 1.
	Probabilities map[int]float64

	// Confidence in [0, 1] summarises how concentrated Probabilities is.
	// Low confidence on a Score usually means the levels are ambiguous or
	// overlapping, or that the state does not contain enough to place it.
	Confidence float64
}

// Kind implements Answer.
func (ScoreAnswer) Kind() QuestionKind { return KindScore }

func (ScoreAnswer) sealed() {}

// Noul returns the probability answer for id.
//
// It reports an error when id is absent or was answered with a different
// primitive, so a caller can tell a real 0 from a missing answer — a
// distinction that matters enormously when the value feeds a threshold.
func (j *Judgment) Noul(id string) (float64, error) {
	a, err := j.answer(id)
	if err != nil {
		return 0, err
	}
	n, ok := a.(NoulAnswer)
	if !ok {
		return 0, fmt.Errorf("%w: %q is a %s, not a noul", ErrAnswerType, id, a.Kind())
	}
	return n.Noul, nil
}

// Choice returns the choice answer for id.
func (j *Judgment) Choice(id string) (ChoiceAnswer, error) {
	a, err := j.answer(id)
	if err != nil {
		return ChoiceAnswer{}, err
	}
	c, ok := a.(ChoiceAnswer)
	if !ok {
		return ChoiceAnswer{}, fmt.Errorf("%w: %q is a %s, not a choice", ErrAnswerType, id, a.Kind())
	}
	return c, nil
}

// Score returns the score answer for id.
func (j *Judgment) Score(id string) (ScoreAnswer, error) {
	a, err := j.answer(id)
	if err != nil {
		return ScoreAnswer{}, err
	}
	s, ok := a.(ScoreAnswer)
	if !ok {
		return ScoreAnswer{}, fmt.Errorf("%w: %q is a %s, not a score", ErrAnswerType, id, a.Kind())
	}
	return s, nil
}

// answer looks up one answer, treating a nil Judgment as "no answers" so a
// caller that ignored an error does not panic on the accessor.
func (j *Judgment) answer(id string) (Answer, error) {
	if j == nil || j.Answers == nil {
		return nil, fmt.Errorf("%w: %q", ErrAnswerMissing, id)
	}
	a, ok := j.Answers[id]
	if !ok || a == nil {
		return nil, fmt.Errorf("%w: %q", ErrAnswerMissing, id)
	}
	return a, nil
}

// sortedKeys returns the map's keys in sorted order, so error messages and
// any size accounting over a question set are deterministic.
func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
