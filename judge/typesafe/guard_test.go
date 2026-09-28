package typesafe_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/judge/typesafe"
)

// These tests run the guard with its real estimator, not the byte-counting
// stub newTestJudge installs: the property under test is the estimator's
// cost on the request path (HAL-1708).

// newRealGuardJudge wires a Judge with the default estimator at srv, and
// records every RequestTrace.
func newRealGuardJudge(t testing.TB, h http.HandlerFunc) (*typesafe.Judge, *[]typesafe.RequestTrace) {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	var mu sync.Mutex
	traces := &[]typesafe.RequestTrace{}
	j, err := typesafe.New(typesafe.Config{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		OnRequest: func(tr typesafe.RequestTrace) {
			mu.Lock()
			*traces = append(*traces, tr)
			mu.Unlock()
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return j, traces
}

// answerAll answers every question the request asked with a Noul.
func answerAll(n int) string {
	var b strings.Builder
	b.WriteString("{")
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, `"p_%d":{"type":"noul","noul":0.5}`, i)
	}
	b.WriteString("}")
	return b.String()
}

// filingPage is dense 10-K prose with figures, about 6,000 characters.
var filingPage = strings.Repeat("Revenue for fiscal 2022 was $12,345 million, up 4.5% year over year, driven by services. ", 66)

// fullBatch is the shape retrieval's full-page read sends: sixteen pages
// in the state, one Noul per page.
func fullBatch(pages int) llmgate.JudgeRequest {
	state := map[string]any{"question": "What was FY2022 revenue?"}
	qs := map[string]llmgate.Question{}
	for i := 0; i < pages; i++ {
		k := fmt.Sprintf("p_%d", i)
		state[k] = map[string]any{"page": i + 1, "text": filingPage[:len(filingPage)*3/4]}
		qs[k] = llmgate.Noul{
			Instructions: fmt.Sprintf("`question` is a question about a document. `%s.text` is the text of page `%s.page`. Does this page contain the specific facts or figures needed to answer the question?", k, k),
			Criteria:     &llmgate.NoulCriteria{True: "The answer is on this page", False: "It is not"},
		}
	}
	return llmgate.JudgeRequest{State: state, Questions: qs}
}

// headBatch is the coarse skim: 120 page heads, 700 characters each.
func headBatch(pages int) llmgate.JudgeRequest {
	state := map[string]any{"question": "What was FY2022 revenue?"}
	qs := map[string]llmgate.Question{}
	for i := 0; i < pages; i++ {
		k := fmt.Sprintf("p_%d", i)
		state[k] = map[string]any{"page": i + 1, "text": filingPage[:700]}
		qs[k] = llmgate.Noul{Instructions: fmt.Sprintf("Could the answer be on page `%s.page`, judging from `%s.text`?", k, k)}
	}
	return llmgate.JudgeRequest{State: state, Questions: qs}
}

func TestGuardOnAFullBatchIsCheap(t *testing.T) {
	req := fullBatch(16)
	j, traces := newRealGuardJudge(t, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, answerAll(len(req.Questions)))
	})
	// Build the encoder outside the measurement: a process pays that once.
	if _, err := typesafe.EstimateTokens("warm"); err != nil {
		t.Fatal(err)
	}
	if _, err := j.Judge(context.Background(), req); err != nil {
		t.Fatalf("Judge: %v", err)
	}
	tr := (*traces)[0]
	if !tr.GuardCounted {
		t.Fatalf("a ~%d-byte state should be past the byte bound and tokenised", tr.StateBytes)
	}
	// Before HAL-1708 this was 17 encoder builds at ~256 ms each: over 4 s.
	// The ceiling is loose so -race and a busy CI box still pass it.
	if tr.Prepare > time.Second {
		t.Errorf("prepare took %v for a 16-page batch; the guard is on the request path and must not dominate it", tr.Prepare)
	}
}

func TestGuardSkipsTheTokenizerWhenBytesProveItFits(t *testing.T) {
	req := headBatch(40)
	j, traces := newRealGuardJudge(t, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, answerAll(len(req.Questions)))
	})
	if _, err := j.Judge(context.Background(), req); err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if tr := (*traces)[0]; tr.GuardCounted {
		t.Errorf("state %d bytes: every token covers a byte, so this provably fits and should not be tokenised", tr.StateBytes)
	}
}

func TestRealGuardStillRejectsAnOversizedState(t *testing.T) {
	var called atomic.Bool
	j, _ := newRealGuardJudge(t, func(w http.ResponseWriter, r *http.Request) {
		called.Store(true)
		writeAnswers(w, `{"q":{"type":"noul","noul":0.5}}`)
	})
	// ~80k tokens of ordinary words: over the 64k total by any tokenizer.
	_, err := j.Judge(context.Background(), llmgate.JudgeRequest{
		State:     strings.Repeat("hello world ", 40_000),
		Questions: map[string]llmgate.Question{"q": llmgate.Noul{Instructions: "urgent?"}},
	})
	if got := llmgate.Classify(err); got != llmgate.ErrClassContextLength {
		t.Fatalf("Classify(%v) = %v, want context_length", err, got)
	}
	if called.Load() {
		t.Error("the oversized request reached the server")
	}
}

func TestRealGuardStillRejectsOneOversizedQuestion(t *testing.T) {
	j, _ := newRealGuardJudge(t, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, `{"big":{"type":"noul","noul":0.5}}`)
	})
	// A 20k-token state is fine alone; a 20k-token question beside it is not.
	_, err := j.Judge(context.Background(), llmgate.JudgeRequest{
		State: strings.Repeat("hello world ", 10_000),
		Questions: map[string]llmgate.Question{
			"big": llmgate.Noul{Instructions: strings.Repeat("hello world ", 10_000)},
		},
	})
	if got := llmgate.Classify(err); got != llmgate.ErrClassContextLength {
		t.Fatalf("Classify(%v) = %v, want context_length", err, got)
	}
	if !strings.Contains(err.Error(), "big") {
		t.Errorf("error %q should name the offending question", err)
	}
}

// Near the ceiling bytes alone cannot decide and the state must be counted:
// ~31k bytes that tokenise to ~5k, beside a question just over 1k bytes,
// fit in tokens though not in bytes.
func TestGuardCountsWhenBytesCannotDecide(t *testing.T) {
	j, traces := newRealGuardJudge(t, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, `{"q":{"type":"noul","noul":0.5}}`)
	})
	_, err := j.Judge(context.Background(), llmgate.JudgeRequest{
		State:     strings.Repeat("hello world ", 2_600),
		Questions: map[string]llmgate.Question{"q": llmgate.Noul{Instructions: strings.Repeat("is it so? ", 120)}},
	})
	if err != nil {
		t.Fatalf("a request that fits in tokens was refused: %v", err)
	}
	if !(*traces)[0].GuardCounted {
		t.Error("the byte bound alone cannot prove this fits; the guard should have counted")
	}
}

func TestOnRequestReportsWhereTheTimeWent(t *testing.T) {
	j, traces := newRealGuardJudge(t, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(20 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"model":"jev-1.13","answers":{"p_0":{"type":"noul","noul":0.9}},"usage":{"input_tokens":321,"output_tokens":0}}`)
	})
	if _, err := j.Judge(context.Background(), headBatch(1)); err != nil {
		t.Fatalf("Judge: %v", err)
	}
	tr := (*traces)[0]
	if tr.Questions != 1 || tr.StatusCode != 200 || tr.InputTokens != 321 || tr.Err != nil {
		t.Errorf("trace = %+v", tr)
	}
	if tr.FirstByte < 20*time.Millisecond {
		t.Errorf("FirstByte = %v, want at least the server's 20ms", tr.FirstByte)
	}
	if tr.Total < tr.FirstByte || tr.Total < tr.Prepare {
		t.Errorf("Total %v must cover Prepare %v and FirstByte %v", tr.Total, tr.Prepare, tr.FirstByte)
	}
	if tr.StateBytes == 0 || tr.BodyBytes <= tr.StateBytes {
		t.Errorf("StateBytes %d, BodyBytes %d", tr.StateBytes, tr.BodyBytes)
	}
	if tr.Model != typesafe.DefaultModel {
		t.Errorf("Model = %q", tr.Model)
	}
}

func TestOnRequestSeesFailures(t *testing.T) {
	j, traces := newRealGuardJudge(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"message":"overloaded"}`, 529)
	})
	_, err := j.Judge(context.Background(), headBatch(1))
	if err == nil {
		t.Fatal("want an error")
	}
	tr := (*traces)[0]
	if tr.Err == nil || tr.StatusCode != 529 {
		t.Errorf("trace = %+v, want the 529 and its error", tr)
	}
}

func TestWarmLeavesAConnectionForTheFirstCall(t *testing.T) {
	var gets atomic.Int32
	j, traces := newRealGuardJudge(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			gets.Add(1)
			http.NotFound(w, r)
			return
		}
		writeAnswers(w, answerAll(1))
	})
	if err := j.Warm(context.Background()); err != nil {
		t.Fatalf("Warm: %v", err)
	}
	if gets.Load() != 1 {
		t.Fatalf("Warm sent %d GETs, want 1", gets.Load())
	}
	if _, err := j.Judge(context.Background(), headBatch(1)); err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if !(*traces)[0].ConnReused {
		t.Error("the first judgment opened a new connection; Warm should have left one in the pool")
	}
}

// The encoder is now shared across goroutines; retrieval sends a level's
// batches concurrently.
func TestEstimateTokensIsSafeConcurrently(t *testing.T) {
	want, err := typesafe.EstimateTokens(filingPage)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 32)
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := typesafe.EstimateTokens(filingPage)
			if err != nil || got != want {
				errs <- fmt.Errorf("got %d, %v; want %d", got, err, want)
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// Before HAL-1708: ~256 ms/op. Acceptance: under 1 ms.
func BenchmarkEstimateTokensShortQuestion(b *testing.B) {
	q := `{"type":"noul","instructions":"Does this page contain the specific facts or figures needed to answer the question?"}`
	typesafe.EstimateTokens(q)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		typesafe.EstimateTokens(q)
	}
}

// The coarse skim's request, end to end against a local stub: guard,
// marshal, one loopback round trip. Acceptance: the guard under 50 ms.
func BenchmarkJudgeHeadBatch120(b *testing.B) {
	req := headBatch(120)
	j, _ := newRealGuardJudge(b, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, answerAll(120))
	})
	typesafe.EstimateTokens("warm")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := j.Judge(context.Background(), req); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkJudgeFullBatch16(b *testing.B) {
	req := fullBatch(16)
	j, _ := newRealGuardJudge(b, func(w http.ResponseWriter, r *http.Request) {
		writeAnswers(w, answerAll(16))
	})
	typesafe.EstimateTokens("warm")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := j.Judge(context.Background(), req); err != nil {
			b.Fatal(err)
		}
	}
}
