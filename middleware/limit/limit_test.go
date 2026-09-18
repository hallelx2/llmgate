package limit

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/middleware/retry"
)

// capacityClient is a provider that serves at most cap calls at once and
// answers 429 to the rest — the shape every real provider has and none
// documents.
type capacityClient struct {
	cap      int
	inflight int32
	peak     int32
	rejected int32
	served   int32
	delay    time.Duration
}

func (c *capacityClient) Complete(ctx context.Context, _ llmgate.Request) (*llmgate.Response, error) {
	n := atomic.AddInt32(&c.inflight, 1)
	defer atomic.AddInt32(&c.inflight, -1)
	for {
		p := atomic.LoadInt32(&c.peak)
		if n <= p || atomic.CompareAndSwapInt32(&c.peak, p, n) {
			break
		}
	}
	if int(n) > c.cap {
		atomic.AddInt32(&c.rejected, 1)
		return nil, llmgate.NewLLMError(llmgate.ProviderOpenAI, 429, "too many requests", nil)
	}
	time.Sleep(c.delay)
	atomic.AddInt32(&c.served, 1)
	return &llmgate.Response{Content: "ok"}, nil
}

func (c *capacityClient) CountTokens(context.Context, string) (int, error) { return 0, nil }

// The property the package exists for: against a provider that admits N
// at once, the limit converges to about N, and — inside retry — no
// request is lost.
func TestLimiterConvergesOnTheProvidersCapacityAndLosesNothing(t *testing.T) {
	prov := &capacityClient{cap: 4, delay: 5 * time.Millisecond}
	var events []Event
	var emu sync.Mutex
	l := New(Config{Initial: 16, Max: 32, SuccessWindow: 5, OnChange: func(e Event) {
		emu.Lock()
		events = append(events, e)
		emu.Unlock()
	}})
	client := retry.New(retry.Config{MaxRetries: 8, BaseDelay: time.Millisecond, MaxDelay: 5 * time.Millisecond})(Client(l)(prov))

	const N = 200
	var wg sync.WaitGroup
	var failed int32
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := client.Complete(context.Background(), llmgate.Request{}); err != nil {
				atomic.AddInt32(&failed, 1)
			}
		}()
	}
	wg.Wait()

	if failed != 0 {
		t.Errorf("%d of %d requests failed past retry; the limiter should have kept them under capacity", failed, N)
	}
	if got := prov.served; got != N {
		t.Errorf("served %d want %d", got, N)
	}
	lim := l.Limit()
	if lim < 2 || lim > 8 {
		t.Errorf("limit %d after the run; want it settled near the provider's capacity of 4", lim)
	}
	emu.Lock()
	defer emu.Unlock()
	var dec, inc int
	for _, e := range events {
		if e.Cause == "rate_limited" {
			dec++
		} else if e.Cause == "success" {
			inc++
		}
	}
	if dec == 0 {
		t.Errorf("starting at 16 against capacity 4, at least one decrease was expected")
	}
	if inc == 0 {
		t.Errorf("after settling, successes should have widened the limit at least once")
	}
}

func TestDecreaseHalvesAndHonoursRetryAfter(t *testing.T) {
	now := time.Unix(1000, 0)
	l := New(Config{Initial: 8, Now: func() time.Time { return now }})
	rel, _ := l.Acquire(context.Background())
	rl := &llmgate.LLMError{Class: llmgate.ErrClassRateLimited, StatusCode: 429, RetryAfterDur: 3 * time.Second}
	rel(rl)
	if l.Limit() != 4 {
		t.Errorf("limit after a 429: %d want 4", l.Limit())
	}
	st := l.Stats()
	if st.PausedUntil.IsZero() || !st.PausedUntil.Equal(now.Add(3*time.Second)) {
		t.Errorf("pause not recorded from Retry-After: %+v", st)
	}
	// While paused, Acquire must not hand out a slot.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, err := l.Acquire(ctx); err == nil {
		t.Errorf("acquired a slot during a provider-requested pause")
	}
	// Pause over: slots flow again.
	now = now.Add(4 * time.Second)
	rel2, err := l.Acquire(context.Background())
	if err != nil {
		t.Fatalf("after the pause: %v", err)
	}
	rel2(nil)
}

func TestSuccessWindowWidensAndFloorCeilingHold(t *testing.T) {
	l := New(Config{Initial: 2, Min: 1, Max: 3, SuccessWindow: 3})
	for i := 0; i < 3; i++ {
		rel, _ := l.Acquire(context.Background())
		rel(nil)
	}
	if l.Limit() != 3 {
		t.Errorf("after a window of successes: %d want 3", l.Limit())
	}
	for i := 0; i < 6; i++ {
		rel, _ := l.Acquire(context.Background())
		rel(nil)
	}
	if l.Limit() != 3 {
		t.Errorf("ceiling: %d want 3", l.Limit())
	}
	transient := &llmgate.LLMError{Class: llmgate.ErrClassTransient, StatusCode: 503}
	for i := 0; i < 5; i++ {
		rel, _ := l.Acquire(context.Background())
		rel(transient)
	}
	if l.Limit() != 1 {
		t.Errorf("floor: %d want 1", l.Limit())
	}
}

func TestNonCapacityErrorsDoNotMoveTheLimit(t *testing.T) {
	l := New(Config{Initial: 4})
	rel, _ := l.Acquire(context.Background())
	rel(&llmgate.LLMError{Class: llmgate.ErrClassAuth, StatusCode: 401})
	rel2, _ := l.Acquire(context.Background())
	rel2(errors.New("bad request"))
	if l.Limit() != 4 {
		t.Errorf("an auth or bad-request error is not about capacity: limit %d", l.Limit())
	}
}

func TestAcquireHonoursContextWhileWaiting(t *testing.T) {
	l := New(Config{Initial: 1})
	rel, _ := l.Acquire(context.Background())
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := l.Acquire(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("waiting past ctx: %v", err)
	}
	rel(nil)
	if l.Stats().Waiting != 0 {
		t.Errorf("a cancelled waiter is still counted: %+v", l.Stats())
	}
}

func TestJudgeWrapperTakesASlotPerBatch(t *testing.T) {
	l := New(Config{Initial: 1})
	var inflight, peak int32
	mock := &llmgate.MockJudge{Respond: func(ctx context.Context, req llmgate.JudgeRequest) (*llmgate.Judgment, error) {
		n := atomic.AddInt32(&inflight, 1)
		defer atomic.AddInt32(&inflight, -1)
		for {
			p := atomic.LoadInt32(&peak)
			if n <= p || atomic.CompareAndSwapInt32(&peak, p, n) {
				break
			}
		}
		time.Sleep(2 * time.Millisecond)
		return &llmgate.Judgment{Model: "mock", Answers: map[string]llmgate.Answer{}}, nil
	}}
	j := Judge(l)(mock)
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := j.Judge(context.Background(), llmgate.JudgeRequest{
				State:     map[string]any{"x": "y"},
				Questions: map[string]llmgate.Question{"q": llmgate.Noul{Instructions: "Is `x` y?", Criteria: &llmgate.NoulCriteria{True: "yes", False: "no"}}},
			})
			if err != nil {
				t.Errorf("judge: %v", err)
			}
		}()
	}
	wg.Wait()
	if peak != 1 {
		t.Errorf("limit 1 let %d batches run at once", peak)
	}
}
