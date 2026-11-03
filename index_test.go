package hnsw

import (
	"cmp"
	"errors"
	"math"
	"math/rand/v2"
	"reflect"
	"runtime"
	"slices"
	"sync"
	"testing"
)

// clustered returns n unit vectors from 256 clusters of Zipf-like sizes. Each cluster is
// a random centre plus a random rank-16 subspace and small isotropic noise, standing in
// for embeddings: low intrinsic dimension, uneven density. It is synthetic.
func clustered(n, d int, seed uint64) [][]float32 {
	const nc, rank = 256, 16
	r := rand.New(rand.NewPCG(seed, 2))
	centres := make([][]float32, nc)
	bases := make([][]float32, nc)
	cdf := make([]float64, nc)
	total := 0.0
	for c := range nc {
		centres[c] = make([]float32, d)
		for i := range centres[c] {
			centres[c][i] = float32(r.NormFloat64())
		}
		if err := normalize(centres[c], centres[c]); err != nil {
			panic(err)
		}
		spread := 0.1 + 0.5*r.Float64()
		bases[c] = make([]float32, rank*d)
		for i := range bases[c] {
			bases[c][i] = float32(r.NormFloat64() * spread / math.Sqrt(float64(d)*rank))
		}
		total += 1 / float64(c+1)
		cdf[c] = total
	}
	noise := 0.02 / math.Sqrt(float64(d))
	out := make([][]float32, n)
	for i := range out {
		u := r.Float64() * total
		c := 0
		for cdf[c] < u {
			c++
		}
		v := slices.Clone(centres[c])
		for j := range rank {
			z := float32(r.NormFloat64())
			for k, b := range bases[c][j*d : (j+1)*d] {
				v[k] += z * b
			}
		}
		for k := range v {
			v[k] += float32(r.NormFloat64() * noise)
		}
		if err := normalize(v, v); err != nil {
			panic(err)
		}
		out[i] = v
	}
	return out
}

// truthDistances returns the exact k nearest distances from each query over base.
func truthDistances(base, queries [][]float32, k int) [][]float32 {
	out := make([][]float32, len(queries))
	var wg sync.WaitGroup
	for w := range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for q := w; q < len(queries); q += runtime.GOMAXPROCS(0) {
				d := make([]float32, len(base))
				for i, v := range base {
					d[i] = distance(queries[q], v)
				}
				slices.Sort(d)
				out[q] = d[:min(k, len(d))]
			}
		}()
	}
	wg.Wait()
	return out
}

// recall is tie-aware recall@k: a result counts when its distance is within the true
// k-th distance plus rounding slack, so equally near points are not marked wrong.
func recall[K Key](t *testing.T, ix *Index[K], queries, truth [][]float32, k, ef int) float64 {
	t.Helper()
	hits := 0
	for i, q := range queries {
		res, err := ix.Search(q, k, SearchOptions{Ef: ef})
		if err != nil {
			t.Fatal(err)
		}
		limit := max(0, truth[i][k-1]) + 2e-6
		for _, r := range res {
			if r.Distance <= limit {
				hits++
			}
		}
	}
	return float64(hits) / float64(k*len(queries))
}

func build(t *testing.T, vecs [][]float32, o Options) *Index[int] {
	t.Helper()
	ix, err := New[int](len(vecs[0]), o)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vecs {
		if err := ix.Add(i, v); err != nil {
			t.Fatal(err)
		}
	}
	return ix
}

func checkGraph(t *testing.T, ix *Index[int]) {
	t.Helper()
	if err := ix.g.verify(); err != nil {
		t.Fatal(err)
	}
	if u := ix.g.unreached(); u != 0 {
		t.Fatalf("%d nodes unreached", u)
	}
}

func TestRecallClustered(t *testing.T) {
	const n, nq, d, k = 20000, 1000, 128, 10
	all := clustered(n+nq, d, 1)
	base, queries := all[:n], all[n:]
	ix := build(t, base, Options{M: 16, Seed: 1})
	checkGraph(t, ix)
	if r := recall(t, ix, queries, truthDistances(base, queries, k), k, 100); r < .999 {
		t.Fatalf("recall@%d at ef=100: %.4f, want >= .999", k, r)
	} else {
		t.Logf("recall@%d at ef=100: %.4f", k, r)
	}
}

// With ef at least n the search reaches every reachable node, so results must be exact.
// M=2 and 3 are left out: they leave some nodes unreachable, as the prototype did too.
func TestSearchExactAtFullBeam(t *testing.T) {
	for _, m := range []int{4, 16} {
		const n, d, k = 600, 24, 20
		all := clustered(n+50, d, uint64(m))
		base, queries := all[:n], all[n:]
		ix := build(t, base, Options{M: m, Seed: 9})
		checkGraph(t, ix)
		truth := truthDistances(base, queries, k)
		for i, q := range queries {
			res, err := ix.Search(q, k, SearchOptions{Ef: n})
			if err != nil {
				t.Fatal(err)
			}
			if len(res) != k {
				t.Fatalf("M=%d: %d results, want %d", m, len(res), k)
			}
			for j, r := range res {
				if r.Distance != max(0, truth[i][j]) {
					t.Fatalf("M=%d query %d rank %d: distance %v, want %v", m, i, j, r.Distance, truth[i][j])
				}
				if got := distance(q, base[r.Key]); max(0, got) != r.Distance {
					t.Fatalf("M=%d: key %d reported at %v, is at %v", m, r.Key, r.Distance, got)
				}
			}
		}
	}
}

// The reference below shares no search, selection or cap code with the index, so a bug
// in those cannot cancel out. It works on plain sorted slices and a map.

func refCompare(a, b item) int { return cmp.Or(cmp.Compare(a.dist, b.dist), cmp.Compare(a.id, b.id)) }

// refLayer is Algorithm 2. It returns W in (distance, ID) order.
func refLayer(g *graph[int], q []float32, ep item, ef, layer int) []item {
	seen := map[uint32]bool{ep.id: true}
	cand, w := []item{ep}, []item{ep}
	for len(cand) > 0 {
		slices.SortFunc(cand, refCompare)
		c := cand[0]
		cand = cand[1:]
		if len(w) >= ef && c.dist > w[len(w)-1].dist {
			break
		}
		for _, e := range g.list(c.id, layer) {
			if seen[e] {
				continue
			}
			seen[e] = true
			x := item{distance(q, g.vec(e)), e}
			if len(w) < ef || x.dist < w[len(w)-1].dist {
				cand = append(cand, x)
				w = append(w, x)
				slices.SortFunc(w, refCompare)
				w = w[:min(len(w), ef)]
			}
		}
	}
	return w
}

// refSelect is Algorithm 4 with a strict comparison.
func refSelect(g *graph[int], cands []item, m int) []item {
	var out []item
	for _, c := range cands {
		if len(out) == m {
			break
		}
		keep := true
		for _, r := range out {
			if distance(g.vec(c.id), g.vec(r.id)) < c.dist {
				keep = false
			}
		}
		if keep {
			out = append(out, c)
		}
	}
	return out
}

// refAdd inserts in place in the paper's order, writing each list as soon as it is
// chosen. Plan-then-publish must build the same graph.
func refAdd(ix *Index[int], key int, v []float32) {
	g := ix.g
	q := make([]float32, g.dim)
	if err := normalize(q, v); err != nil {
		panic(err)
	}
	id, lv := g.n, ix.levelOf(g.n)
	capOf := func(layer int) int {
		if layer == 0 {
			return 2 * g.m
		}
		return g.m
	}
	g.appendVec(q)
	g.level = append(g.level, uint8(lv))
	g.upOff = append(g.upOff, uint32(len(g.up)/(g.m+1)))
	g.up = append(g.up, make([]uint32, lv*(g.m+1))...)
	g.adj0 = append(g.adj0, make([]uint32, 2*g.m+1)...)
	g.keys = append(g.keys, key)
	g.byKey[key] = id
	if id > 0 {
		ep := item{distance(q, g.vec(g.entry)), g.entry}
		for layer := int(g.top); layer > lv; layer-- {
			ep = refLayer(g, q, ep, 1, layer)[0]
		}
		for layer := min(int(g.top), lv); layer >= 0; layer-- {
			cands := refLayer(g, q, ep, ix.efc, layer)
			ep = cands[0]
			sel := refSelect(g, cands, g.m)
			r := g.record(id, layer)
			for i, s := range sel {
				r[1+i] = s.id
			}
			r[0] = uint32(len(sel))
			for _, s := range sel {
				r := g.record(s.id, layer)
				if int(r[0]) < capOf(layer) {
					r[1+r[0]] = id
					r[0]++
					continue
				}
				var links []item
				for _, x := range r[1 : 1+r[0]] {
					links = append(links, item{distance(g.vec(s.id), g.vec(x)), x})
				}
				links = append(links, item{distance(g.vec(s.id), q), id})
				slices.SortFunc(links, refCompare)
				kept := refSelect(g, links, capOf(layer))
				clear(r)
				for i, x := range kept {
					r[1+i] = x.id
				}
				r[0] = uint32(len(kept))
			}
		}
	}
	if id == 0 || lv > int(g.top) {
		g.entry, g.top = id, uint8(lv)
	}
	g.n++
	g.live++
}

// lattice returns n vectors with small integer coordinates, so many distances tie
// exactly and tie handling has to match too. Zero vectors are skipped.
func lattice(n, d int, seed uint64) [][]float32 {
	s := splitmix(seed)
	out := make([][]float32, 0, n)
	for len(out) < n {
		v := make([]float32, d)
		for i := range v {
			v[i] = float32(int(s.next()%5) - 2)
		}
		if normalize(v, v) == nil {
			out = append(out, v)
		}
	}
	return out
}

// referenceData is clustered data plus tie-heavy lattice data.
var referenceData = map[string]func() [][]float32{
	"clustered": func() [][]float32 { return clustered(3200, 16, 5) },
	"lattice":   func() [][]float32 { return lattice(3200, 4, 5) },
}

func TestPlanMatchesReferenceInserter(t *testing.T) {
	for name, data := range referenceData {
		vecs := data()[:3000]
		for _, m := range []int{2, 4, 16} {
			got := build(t, vecs, Options{M: m, EfConstruction: 40, Seed: 3})
			want, err := New[int](len(vecs[0]), Options{M: m, EfConstruction: 40, Seed: 3})
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range vecs {
				refAdd(want, i, v)
			}
			if err := want.g.verify(); err != nil {
				t.Fatal(err)
			}
			if want.g.top == 0 {
				t.Fatalf("%s M=%d: no upper layers, so they went untested", name, m)
			}
			if !reflect.DeepEqual(got.g, want.g) {
				t.Fatalf("%s M=%d: graphs differ", name, m)
			}
		}
	}
}

// Search is Algorithm 5: descend with ef=1, then search layer 0 with min(max(ef,k), live).
func TestSearchMatchesReference(t *testing.T) {
	const n, k = 3000, 10
	for name, data := range referenceData {
		all := data()
		ix := build(t, all[:n], Options{M: 4, EfConstruction: 40, Seed: 3})
		g := ix.g
		for _, ef := range []int{1, 10, 50} {
			for _, q := range all[n:] {
				ep := item{distance(q, g.vec(g.entry)), g.entry}
				for layer := int(g.top); layer > 0; layer-- {
					ep = refLayer(g, q, ep, 1, layer)[0]
				}
				w := refLayer(g, q, ep, min(max(ef, k), int(g.live)), 0)
				w = w[:min(k, len(w))]
				want := make([]Result[int], len(w))
				for i, x := range w {
					want[i] = Result[int]{g.keys[x.id], max(0, x.dist)}
				}
				got, err := ix.Search(q, k, SearchOptions{Ef: ef})
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("%s ef=%d: got %v, want %v", name, ef, got, want)
				}
			}
		}
	}
}

func TestSelectIsStrict(t *testing.T) {
	ix, err := New[int](3, Options{M: 2})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range [][]float32{{0.6, 0, 0.8}, {0, 1, 0}, {1, 0, 0}} {
		if err := ix.Add(i, v); err != nil {
			t.Fatal(err)
		}
	}
	// For node 2, node 0 is at 0.4 and node 1 at exactly 1, and node 1 is exactly 1 from
	// node 0: not strictly closer, so both stay.
	if got := ix.g.list(2, 0); len(got) != 2 {
		t.Fatalf("links %v, want both", got)
	}
}

// Stored vectors can self-match at a slightly negative raw distance.
func TestSearchClampsDistance(t *testing.T) {
	vecs := clustered(600, 24, 4)
	ix := build(t, vecs, Options{M: 4, Seed: 9})
	negative := 0
	for _, v := range vecs {
		if distance(v, v) < 0 {
			negative++
		}
		res, err := ix.Search(v, 1, SearchOptions{Ef: len(vecs)})
		if err != nil || res[0].Distance < 0 {
			t.Fatalf("%v, %v", res, err)
		}
	}
	if negative == 0 {
		t.Fatal("no negative self-distance in the data, so the clamp went untested")
	}
}

func TestLevelDistribution(t *testing.T) {
	for _, m := range []int{2, 16} {
		ix, err := New[int](1, Options{M: m, Seed: 77})
		if err != nil {
			t.Fatal(err)
		}
		const n = 1 << 20
		atLeast := make([]int, maxLevel+2)
		for id := range uint32(n) {
			lv := ix.levelOf(id)
			if lv < 0 || lv > maxLevel {
				t.Fatalf("level %d", lv)
			}
			for l := 0; l <= lv; l++ {
				atLeast[l]++
			}
		}
		// P(level >= l) = M^-l. Allow five standard deviations of a binomial count.
		for l := 1; l <= 5; l++ {
			p := math.Pow(float64(m), -float64(l))
			want, sd := n*p, math.Sqrt(n*p*(1-p))
			if want < 100 {
				break
			}
			if math.Abs(float64(atLeast[l])-want) > 5*sd {
				t.Errorf("M=%d: %d nodes at level >= %d, want %.0f ± %.0f", m, atLeast[l], l, want, 5*sd)
			}
		}
	}
}

func TestNewValidatesOptions(t *testing.T) {
	for _, tc := range []struct {
		dim int
		o   Options
		ok  bool
	}{
		{1, Options{}, true},
		{maxDim, Options{}, true},
		{0, Options{}, false},
		{-1, Options{}, false},
		{maxDim + 1, Options{}, false},
		{8, Options{M: 2}, true},
		{8, Options{M: 512}, true},
		{8, Options{M: 300}, true}, // the default EfConstruction follows M
		{8, Options{M: 1}, false},
		{8, Options{M: -1}, false},
		{8, Options{M: 513}, false},
		{8, Options{M: 16, EfConstruction: 16}, true},
		{8, Options{M: 16, EfConstruction: 15}, false},
		{8, Options{EfConstruction: -1}, false},
		{8, Options{EfConstruction: maxBeam}, true},
		{8, Options{EfConstruction: maxBeam + 1}, false},
		{8, Options{Ef: 1}, true},
		{8, Options{Ef: maxBeam}, true},
		{8, Options{Ef: -1}, false},
		{8, Options{Ef: maxBeam + 1}, false},
	} {
		_, err := New[int](tc.dim, tc.o)
		if tc.ok && err != nil {
			t.Errorf("dim %d %+v: %v", tc.dim, tc.o, err)
		}
		if !tc.ok && !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("dim %d %+v: got %v, want ErrInvalidArgument", tc.dim, tc.o, err)
		}
	}
}

func TestLargeMAlone(t *testing.T) {
	ix := build(t, clustered(300, 8, 2), Options{M: 300})
	checkGraph(t, ix)
	if ix.efc != 300 {
		t.Fatalf("EfConstruction %d, want 300", ix.efc)
	}
}

func TestBadInput(t *testing.T) {
	ix, err := New[string](3, Options{})
	if err != nil {
		t.Fatal(err)
	}
	nan := float32(math.NaN())
	for _, tc := range []struct {
		v    []float32
		want error
	}{
		{[]float32{1, 2}, ErrDimension},
		{[]float32{1, 2, 3, 4}, ErrDimension},
		{nil, ErrDimension},
		{[]float32{0, 0, 0}, ErrInvalidVector},
		{[]float32{1, nan, 0}, ErrInvalidVector},
		{[]float32{1, float32(math.Inf(-1)), 0}, ErrInvalidVector},
	} {
		if err := ix.Add("a", tc.v); !errors.Is(err, tc.want) {
			t.Errorf("Add %v: got %v, want %v", tc.v, err, tc.want)
		}
		if _, err := ix.Search(tc.v, 1, SearchOptions{}); !errors.Is(err, tc.want) {
			t.Errorf("Search %v: got %v, want %v", tc.v, err, tc.want)
		}
	}
	if ix.Len() != 0 || ix.g.n != 0 {
		t.Fatal("a failed Add changed the index")
	}
	if err := ix.Add("a", []float32{1, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := ix.Add("a", []float32{0, 1, 0}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("existing key: got %v", err)
	}
	if ix.Len() != 1 {
		t.Fatalf("Len %d", ix.Len())
	}
	q := []float32{1, 0, 0}
	for _, tc := range []struct {
		k, ef int
		ok    bool
	}{
		{1, 0, true},
		{maxBeam, 0, true},
		{1, maxBeam, true},
		{0, 0, false},
		{-1, 0, false},
		{maxBeam + 1, 0, false},
		{1, -1, false},
		{1, maxBeam + 1, false},
	} {
		_, err := ix.Search(q, tc.k, SearchOptions{Ef: tc.ef})
		if tc.ok && err != nil {
			t.Errorf("k=%d ef=%d: %v", tc.k, tc.ef, err)
		}
		if !tc.ok && !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("k=%d ef=%d: got %v, want ErrInvalidArgument", tc.k, tc.ef, err)
		}
	}
}

func TestZeroIndex(t *testing.T) {
	var ix Index[int]
	if err := ix.Add(1, []float32{1}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("Add: %v", err)
	}
	if _, err := ix.Search([]float32{1}, 1, SearchOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("Search: %v", err)
	}
	if ix.Len() != 0 {
		t.Error("Len")
	}
}

func TestSmallIndexes(t *testing.T) {
	ix, err := New[[16]byte](2, Options{M: 2})
	if err != nil {
		t.Fatal(err)
	}
	if res, err := ix.Search([]float32{1, 0}, 5, SearchOptions{}); res != nil || err != nil {
		t.Fatalf("empty index: %v, %v", res, err)
	}
	a, b := [16]byte{1}, [16]byte{2}
	if err := ix.Add(a, []float32{3, 4}); err != nil {
		t.Fatal(err)
	}
	res, err := ix.Search([]float32{3, 4}, 5, SearchOptions{})
	if err != nil || len(res) != 1 || res[0].Key != a || res[0].Distance != 0 {
		t.Fatalf("one node: %v, %v", res, err)
	}
	if err := ix.Add(b, []float32{-4, 3}); err != nil {
		t.Fatal(err)
	}
	res, err = ix.Search([]float32{-1, 1}, 5, SearchOptions{Ef: 1})
	if err != nil || len(res) != 2 || res[0].Key != b || res[1].Key != a || res[0].Distance > res[1].Distance {
		t.Fatalf("two nodes: %v, %v", res, err)
	}
	if err := ix.g.verify(); err != nil {
		t.Fatal(err)
	}
}

func TestSearchDoesNotAliasInput(t *testing.T) {
	ix := build(t, clustered(200, 8, 4), Options{})
	q := slices.Clone(ix.g.vec(7))
	q[0] *= 3 // not unit, so Search must normalize a copy
	before := slices.Clone(q)
	res, err := ix.Search(q, 1, SearchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(q, before) {
		t.Fatal("Search modified the query")
	}
	if len(res) != 1 {
		t.Fatalf("%v", res)
	}
	v := slices.Clone(ix.g.vec(3))
	if err := ix.Add(1000, v); err != nil {
		t.Fatal(err)
	}
	v[0] = 42
	if err := ix.g.verify(); err != nil {
		t.Fatalf("Add kept the caller's slice: %v", err)
	}
}

func TestConcurrentSearches(t *testing.T) {
	const n, d = 2000, 16
	all := clustered(n+64, d, 6)
	ix := build(t, all[:n], Options{})
	want := make([][]Result[int], 64)
	for i, q := range all[n:] {
		var err error
		if want[i], err = ix.Search(q, 10, SearchOptions{}); err != nil {
			t.Fatal(err)
		}
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i, q := range all[n:] {
				got, err := ix.Search(q, 10, SearchOptions{})
				if err != nil || !reflect.DeepEqual(got, want[i]) {
					t.Errorf("query %d: %v, %v", i, got, err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestVerifyCatchesCorruption(t *testing.T) {
	for name, corrupt := range map[string]func(g *graph[int]){
		"self loop":    func(g *graph[int]) { g.adj0[1] = 0 },
		"out of range": func(g *graph[int]) { g.adj0[1] = g.n },
		"over cap":     func(g *graph[int]) { g.adj0[0] = uint32(2*g.m + 1) },
		"dirty unused slot": func(g *graph[int]) {
			for id := range g.n {
				if r := g.record(id, 0); int(r[0]) < 2*g.m {
					r[len(r)-1] = 1
					return
				}
			}
		},
		"short level":        func(g *graph[int]) { g.level = g.level[:len(g.level)-1] },
		"short upper offset": func(g *graph[int]) { g.upOff = g.upOff[:len(g.upOff)-1] },
		"short keys":         func(g *graph[int]) { g.keys = g.keys[:len(g.keys)-1] },
		"extra chunk":        func(g *graph[int]) { g.vecs = append(g.vecs, nil) },
		"missing chunk":      func(g *graph[int]) { g.vecs = g.vecs[:0] },
		"short chunk":        func(g *graph[int]) { g.vecs[0] = g.vecs[0][:len(g.vecs[0])-g.dim] },
		"entry out of range": func(g *graph[int]) { g.entry = g.n },
		"level above top": func(g *graph[int]) {
			last := g.n - 1 // the last node keeps every other offset valid
			if g.level[last] != 0 {
				panic("last node is not on layer 0")
			}
			g.level[last] = g.top + 1
			g.up = append(g.up, make([]uint32, int(g.top+1)*(g.m+1))...)
		},
		"repeated link": func(g *graph[int]) {
			r := g.record(0, 0)
			r[2] = r[1]
		},
		"entry below top": func(g *graph[int]) { g.top++ },
		"entry missing":   func(g *graph[int]) { g.entry = noEntry },
		"unnormalized":    func(g *graph[int]) { g.vec(3)[0] *= 2 },
		"negative zero":   func(g *graph[int]) { g.vec(3)[0] = float32(math.Copysign(0, -1)) },
		"NaN":             func(g *graph[int]) { g.vec(3)[0] = float32(math.NaN()) },
		"short adjacency": func(g *graph[int]) { g.adj0 = g.adj0[:len(g.adj0)-1] },
		"short upper":     func(g *graph[int]) { g.up = g.up[:len(g.up)-1] },
		"wrong upper offset": func(g *graph[int]) {
			for id, lv := range g.level {
				if lv > 0 {
					g.upOff[id]++
					return
				}
			}
		},
		"link above level": func(g *graph[int]) {
			for id, lv := range g.level {
				if lv == 0 {
					r := g.record(g.entry, 1)
					r[1+r[0]-1] = uint32(id)
					return
				}
			}
		},
	} {
		ix := build(t, clustered(500, 8, 8), Options{M: 4})
		if err := ix.g.verify(); err != nil {
			t.Fatal(err)
		}
		corrupt(ix.g)
		if err := ix.g.verify(); !errors.Is(err, ErrCorrupt) {
			t.Errorf("%s: got %v, want ErrCorrupt", name, err)
		}
	}
	for name, corrupt := range map[string]func(g *graph[int]){
		"empty with entry": func(g *graph[int]) { g.entry = 0 },
		"empty with top":   func(g *graph[int]) { g.top = 1 },
	} {
		g := newGraph[int](8, 4)
		corrupt(g)
		if err := g.verify(); !errors.Is(err, ErrCorrupt) {
			t.Errorf("%s: got %v, want ErrCorrupt", name, err)
		}
	}
}

func TestUnreachedCounts(t *testing.T) {
	ix := build(t, clustered(50, 8, 1), Options{M: 4})
	clear(ix.g.record(ix.g.entry, 0)) // the entry keeps no layer-0 links, so only it is reached
	if u := ix.g.unreached(); u != int(ix.g.n)-1 {
		t.Fatalf("unreached %d, want %d", u, ix.g.n-1)
	}
}

// unsplitmix inverts splitmix64, so a test can pick the seed that gives an ID any hash.
func unsplitmix(y uint64) uint64 {
	unxorshift := func(x uint64, s uint) uint64 {
		r := x
		for range 64/int(s) + 1 {
			r = x ^ (r >> s)
		}
		return r
	}
	inverse := func(a uint64) uint64 { // Newton's iteration for the inverse mod 2^64
		x := a
		for range 6 {
			x *= 2 - a*x
		}
		return x
	}
	y = unxorshift(y, 31)
	y *= inverse(0x94d049bb133111eb)
	y = unxorshift(y, 27)
	y *= inverse(0xbf58476d1ce4e5b9)
	y = unxorshift(y, 30)
	return y - 0x9e3779b97f4a7c15
}

func TestLevelExtremes(t *testing.T) {
	if splitmix64(unsplitmix(0)) != 0 {
		t.Fatal("unsplitmix does not invert splitmix64")
	}
	// A hash of 0 gives the smallest u, 2^-53: level 53 for M=2, capped to 31, and 13 for M=16.
	for _, tc := range []struct{ m, want int }{{2, maxLevel}, {16, 13}} {
		ix, err := New[int](1, Options{M: tc.m})
		if err != nil {
			t.Fatal(err)
		}
		ix.seed = unsplitmix(0) - 5 // so ID 5 hashes to 0
		if got := ix.levelOf(5); got != tc.want {
			t.Errorf("M=%d: level %d, want %d", tc.m, got, tc.want)
		}
	}
}

func TestDefaults(t *testing.T) {
	ix, err := New[int](8, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if ix.g.m != 16 || ix.efc != 200 || ix.ef != 100 {
		t.Fatalf("M=%d EfConstruction=%d Ef=%d, want 16, 200, 100", ix.g.m, ix.efc, ix.ef)
	}
}

func TestErrFull(t *testing.T) {
	ix, err := New[int](2, Options{})
	if err != nil {
		t.Fatal(err)
	}
	ix.g.n = maxNodes // the check runs before anything reads the arrays
	if err := ix.Add(1, []float32{1, 0}); !errors.Is(err, ErrFull) {
		t.Fatalf("got %v, want ErrFull", err)
	}
}
