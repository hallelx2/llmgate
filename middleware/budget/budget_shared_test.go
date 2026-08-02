package budget_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/hallelx2/llmgate"
	"github.com/hallelx2/llmgate/middleware/budget"
)

// tagged is a client that reports which instance handled a call, so a
// test can tell whether the middleware actually wrapped the client it
// was handed.
type tagged struct {
	name string
	cost float64
	mu   sync.Mutex
	n    int
}

func (c *tagged) Complete(context.Context, llmgate.Request) (*llmgate.Response, error) {
	c.mu.Lock()
	c.n++
	c.mu.Unlock()
	return &llmgate.Response{
		Content: c.name,
		Usage:   llmgate.Usage{CostUSD: c.cost, Priced: true},
	}, nil
}

func (c *tagged) CountTokens(context.Context, string) (int, error) { return 0, nil }

func (c *tagged) calls() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.n
}

// TestMiddlewareWrapsEachClientSeparately is the HAL-526 regression.
//
// New used to build one budgetClient and rewrite its inner pointer on
// every application, so wrapping two clients returned the *same* object
// pointing at whichever was wrapped last. A router composed the way the
// README recommends would then send every request to its fallback and
// never call the primary — silently, with no error.
func TestMiddlewareWrapsEachClientSeparately(t *testing.T) {
	primary := &tagged{name: "primary", cost: 0.01}
	fallback := &tagged{name: "fallback", cost: 0.01}

	mw := budget.New(budget.Config{DailyUSD: 100})
	wrappedPrimary := mw(primary)
	wrappedFallback := mw(fallback)

	resp, err := wrappedPrimary.Complete(context.Background(), llmgate.Request{})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "primary" {
		t.Errorf("wrapped primary reached %q, want primary", resp.Content)
	}

	resp, err = wrappedFallback.Complete(context.Background(), llmgate.Request{})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "fallback" {
		t.Errorf("wrapped fallback reached %q, want fallback", resp.Content)
	}

	if primary.calls() != 1 || fallback.calls() != 1 {
		t.Errorf("calls: primary=%d fallback=%d, want 1 each", primary.calls(), fallback.calls())
	}
}

// TestSharedLedgerAcrossClients: separate wrappers, one budget. Spending
// through either must draw down the same counters.
func TestSharedLedgerAcrossClients(t *testing.T) {
	a := &tagged{name: "a", cost: 0.6}
	b := &tagged{name: "b", cost: 0.6}

	mw := budget.New(budget.Config{TotalUSD: 1.0})
	wa, wb := mw(a), mw(b)

	if _, err := wa.Complete(context.Background(), llmgate.Request{}); err != nil {
		t.Fatalf("first call: %v", err)
	}
	// 0.60 spent against a 1.00 cap; the next call through the *other*
	// wrapper must see it.
	if _, err := wb.Complete(context.Background(), llmgate.Request{}); err != nil {
		t.Fatalf("second call: %v", err)
	}
	if _, err := wb.Complete(context.Background(), llmgate.Request{}); !errors.Is(err, budget.ErrExceeded) {
		t.Fatalf("third call err = %v, want ErrExceeded — the ledger must be shared", err)
	}
}

// TestLedgerObservable: spend has to be readable, or a dashboard and a
// control plane have nothing to show.
func TestLedgerObservable(t *testing.T) {
	l := budget.NewLedger(budget.Config{DailyUSD: 10, TotalUSD: 100})
	c := budget.NewWithLedger(l)(&tagged{name: "c", cost: 2.5})

	if _, err := c.Complete(context.Background(), llmgate.Request{}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	daily, total := l.Spent()
	if daily != 2.5 || total != 2.5 {
		t.Errorf("Spent = (%v, %v), want (2.5, 2.5)", daily, total)
	}
	rd, rt := l.Remaining()
	if rd != 7.5 || rt != 97.5 {
		t.Errorf("Remaining = (%v, %v), want (7.5, 97.5)", rd, rt)
	}
}

// TestUnlimitedCapReportsInfinite: no cap means no headroom to run out
// of, which must not read as zero remaining.
func TestUnlimitedCapReportsInfinite(t *testing.T) {
	l := budget.NewLedger(budget.Config{})
	rd, rt := l.Remaining()
	if rd <= 1e300 || rt <= 1e300 {
		t.Errorf("Remaining = (%v, %v), want +Inf for unset caps", rd, rt)
	}
}

// TestRefuseUnpriced closes the hole where a budget over unpriced models
// is silently unlimited: an unpriced call debits 0, so the cap never
// trips however much is really spent.
func TestRefuseUnpriced(t *testing.T) {
	unpriced := &unpricedClient{}

	permissive := budget.New(budget.Config{TotalUSD: 1})(unpriced)
	if _, err := permissive.Complete(context.Background(), llmgate.Request{}); err != nil {
		t.Fatalf("default config should tolerate unpriced calls: %v", err)
	}

	strict := budget.New(budget.Config{TotalUSD: 1, RefuseUnpriced: true})(unpriced)
	if _, err := strict.Complete(context.Background(), llmgate.Request{}); !errors.Is(err, budget.ErrUnpriced) {
		t.Fatalf("err = %v, want ErrUnpriced when the cost cannot be counted", err)
	}
}

type unpricedClient struct{}

func (unpricedClient) Complete(context.Context, llmgate.Request) (*llmgate.Response, error) {
	return &llmgate.Response{Usage: llmgate.Usage{CostUSD: 0, Priced: false}}, nil
}
func (unpricedClient) CountTokens(context.Context, string) (int, error) { return 0, nil }

// TestConcurrentSpendIsRaceFree exercises the ledger under the kind of
// parallelism the engine actually runs at.
func TestConcurrentSpendIsRaceFree(t *testing.T) {
	l := budget.NewLedger(budget.Config{TotalUSD: 1e9})
	c := budget.NewWithLedger(l)(&tagged{name: "x", cost: 0.001})

	var wg sync.WaitGroup
	for range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = c.Complete(context.Background(), llmgate.Request{})
		}()
	}
	wg.Wait()

	_, total := l.Spent()
	if want := 0.05; total < want*0.99 || total > want*1.01 {
		t.Errorf("total spend = %v, want ~%v after 50 calls", total, want)
	}
}
