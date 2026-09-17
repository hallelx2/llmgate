package typesafe_test

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/hallelx2/llmgate"
)

// Strictness at the decode boundary. Every case here is one where a
// permissive decoder produces a plausible-looking zero that a caller would
// then threshold against — the failure mode that does not announce itself.

// A partial 200 must fail where it happens. Left alone it is silent: the
// judgment looks fine, and the gap only surfaces later as an
// ErrAnswerMissing from an accessor, far from the response that caused it.
func TestPartialResponseIsRejected(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		// Two questions asked, one answered.
		writeAnswers(w, `{"a":{"type":"noul","noul":0.5}}`)
	})

	_, err := j.Judge(context.Background(), llmgate.JudgeRequest{
		State: "s",
		Questions: map[string]llmgate.Question{
			"a": llmgate.Noul{Instructions: "a?"},
			"b": llmgate.Noul{Instructions: "b?"},
		},
	})
	if err == nil {
		t.Fatal("Judge() = nil error, want a partial-response failure")
	}
	if !strings.Contains(err.Error(), "b") {
		t.Errorf("error = %q, want it to name the unanswered question", err)
	}
}

// A missing required field must not decode to a zero. An absent "noul"
// becoming 0.0 does not read as an error downstream — it reads as a
// confident "definitely no".
func TestAnswerWithMissingFieldsIsRejected(t *testing.T) {
	cases := []struct {
		name    string
		answers string
	}{
		{"noul with no value", `{"is_urgent":{"type":"noul"}}`},
		{"choice with no selection", `{"is_urgent":{"type":"choice","probabilities":{"a":1},"confidence":0.9}}`},
		{"choice with no distribution", `{"is_urgent":{"type":"choice","choice":"a","confidence":0.9}}`},
		{"choice with no confidence", `{"is_urgent":{"type":"choice","choice":"a","probabilities":{"a":1}}}`},
		{"score with no value", `{"is_urgent":{"type":"score","probabilities":{"0":1},"confidence":0.9}}`},
		{"score with no distribution", `{"is_urgent":{"type":"score","score":1,"confidence":0.9}}`},
		{"score with no confidence", `{"is_urgent":{"type":"score","score":1,"probabilities":{"0":1}}}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
				writeAnswers(w, tc.answers)
			})
			if _, err := j.Judge(context.Background(), batch()); err == nil {
				t.Error("Judge() = nil error, want a malformed-answer failure")
			}
		})
	}
}

// A probability outside [0,1] means the body is not what it claims to be.
// Passing it through hands a caller a "probability" of 7 to threshold.
func TestOutOfRangeProbabilityIsRejected(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, `{"is_urgent":{"type":"noul","noul":7}}`)
	})

	_, err := j.Judge(context.Background(), batch())
	if err == nil {
		t.Fatal("Judge() = nil error, want an out-of-range failure")
	}
	if !strings.Contains(err.Error(), "[0,1]") {
		t.Errorf("error = %q, want it to name the valid range", err)
	}
}

// A real zero is not a missing count. An explicit
// {"input_tokens":0,"output_tokens":0} has been reported, and calling that
// unreported is the HAL-531 conflation pointing the other way.
func TestExplicitZeroTokensCountAsReported(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"model":"jev-1.13.0","answers":{"is_urgent":{"type":"noul","noul":0.5}},"usage":{"input_tokens":0,"output_tokens":0}}`)
	})

	got, err := j.Judge(context.Background(), batch())
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if !got.Usage.TokensReported {
		t.Error("TokensReported = false for an explicit zero; the provider did report")
	}
}

// An absent usage object is genuinely unreported, and must not be priced.
// A $0 with Priced=true asserts the call was free, which it never is.
func TestAbsentUsageIsNotPriced(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"model":"jev-1.13.0","answers":{"is_urgent":{"type":"noul","noul":0.5}}}`)
	})

	got, err := j.Judge(context.Background(), batch())
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if got.Usage.TokensReported {
		t.Error("TokensReported = true with no usage object at all")
	}
	if got.Usage.Priced {
		t.Error("Priced = true with no token counts; that asserts a free call")
	}
}

func TestReportedUsageIsPriced(t *testing.T) {
	j := newTestJudge(t, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, `{"is_urgent":{"type":"noul","noul":0.5}}`)
	})

	got, err := j.Judge(context.Background(), batch())
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if !got.Usage.Priced {
		t.Fatal("Priced = false for jev-1.13.0, which is in the price book")
	}
	if got.Usage.CostUSD <= 0 {
		t.Errorf("CostUSD = %v, want a positive cost for 312 input tokens", got.Usage.CostUSD)
	}
}
