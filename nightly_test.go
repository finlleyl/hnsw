package hnsw

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"strconv"
	"testing"
)

func TestNightlyRecallClustered(t *testing.T) {
	if testing.Short() {
		t.Skip("nightly")
	}
	const n, nq, k = 100000, 1000, 10
	configs := []struct {
		d, maxUnreached int
		want            [3]float64
	}{
		{128, 0, [3]float64{.985, .992, .995}},
		{768, 1, [3]float64{.982, .990, .995}},
	}
	for _, tc := range configs {
		for _, seed := range []uint64{101, 202, 303} {
			t.Run(fmt.Sprintf("d=%d/seed=%d", tc.d, seed), func(t *testing.T) {
				all := clustered(n+nq, tc.d, seed)
				base, queries := all[:n], all[n:]
				ix := build(t, base, Options{M: 16, Seed: seed})
				if err := ix.g.verify(); err != nil {
					t.Fatal(err)
				}
				u := ix.g.unreached()
				t.Logf("unreached: %d, limit %d", u, tc.maxUnreached)
				if u > tc.maxUnreached {
					t.Errorf("%d nodes unreached, want <= %d", u, tc.maxUnreached)
				}
				truth := truthDistances(base, queries, k)
				for i, ef := range []int{40, 80, 160} {
					r := recall(t, ix, queries, truth, k, ef)
					t.Logf("recall@10 at ef=%d: %.4f", ef, r)
					if r < tc.want[i] {
						t.Errorf("recall@10 at ef=%d: %.4f, want >= %.3f", ef, r, tc.want[i])
					}
				}
			})
		}
	}
}

func TestNightlyGaussianCanary(t *testing.T) {
	if testing.Short() {
		t.Skip("nightly")
	}
	const n, nq, d, k = 100000, 1000, 128, 10
	r := rand.New(rand.NewPCG(7, 2))
	all := make([][]float32, n+nq)
	for i := range all {
		v := make([]float32, d)
		for j := range v {
			v[j] = float32(r.NormFloat64())
		}
		if err := normalize(v, v); err != nil {
			t.Fatal(err)
		}
		all[i] = v
	}
	base, queries := all[:n], all[n:]
	ix := build(t, base, Options{M: 16, Seed: 7})
	if err := ix.g.verify(); err != nil {
		t.Fatal(err)
	}
	u := ix.g.unreached()
	t.Logf("unreached: %d", u)
	if u != 0 {
		t.Errorf("%d nodes unreached", u)
	}
	got := recall(t, ix, queries, truthDistances(base, queries, k), k, 640)
	t.Logf("gaussian recall@10 at ef=640: %.4f", got)
	if got < .75 {
		t.Fatalf("gaussian recall@10 at ef=640: %.4f, want >= .75", got)
	}
}

func TestBytesPerVector(t *testing.T) {
	if testing.Short() {
		t.Skip("nightly")
	}
	if strconv.IntSize < 64 {
		t.Skip("the 1M-vector measurement requires a 64-bit platform")
	}
	const n, d, batch = 1000000, 768, 10000
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	ix, err := New[uint64](d, Options{M: 16, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	var vecs [][]float32
	for key := range n {
		if key%batch == 0 {
			vecs = clustered(batch, d, uint64(7+key/batch))
		}
		if err := ix.Add(uint64(key), vecs[key%batch]); err != nil {
			t.Fatal(err)
		}
	}
	vecs = nil
	runtime.GC()
	runtime.ReadMemStats(&after)
	runtime.KeepAlive(ix)
	stats := ix.Stats()
	if stats.Keys != n {
		t.Fatalf("got %d keys, want %d", stats.Keys, n)
	}
	t.Logf("built %d keys on %d nodes", stats.Keys, stats.Nodes)
	got := float64(after.HeapAlloc-before.HeapAlloc) / n
	want := float64(4*d + 226)
	t.Logf("live heap: %.1f bytes/vector, want %.0f within 10%%", got, want)
	if got < .9*want || got > 1.1*want {
		t.Fatalf("live heap: %.1f bytes/vector, want %.0f within 10%%", got, want)
	}
}
