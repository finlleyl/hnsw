package hnsw

import (
	"cmp"
	"errors"
	"maps"
	"math"
	"slices"
	"strings"
	"testing"
	"unsafe"
)

type (
	namedString string
	namedInt    int
	namedInt32  int32
	namedInt64  int64
	namedUint   uint
	namedUint32 uint32
	namedUint64 uint64
	namedUUID   [16]byte
)

// checkKeys runs the key lifecycle for one key type: add, find, replace, delete.
func checkKeys[K Key](t *testing.T, keys [3]K) {
	t.Helper()
	ix, err := New[K](2, Options{M: 2})
	if err != nil {
		t.Fatal(err)
	}
	vecs := [][]float32{{1, 0}, {0, 1}, {-1, 0}}
	for i, k := range keys {
		if err := ix.Add(k, vecs[i]); err != nil {
			t.Fatal(err)
		}
	}
	for i, k := range keys {
		res, err := ix.Search(vecs[i], 1, SearchOptions{})
		if err != nil || len(res) != 1 || res[0].Key != k || !ix.Contains(k) {
			t.Fatalf("key %v: %v, %v", k, res, err)
		}
	}
	if err := ix.Add(keys[0], []float32{0, -1}); err != nil {
		t.Fatal(err)
	}
	if res, _ := ix.Search([]float32{0, -1}, 1, SearchOptions{}); res[0].Key != keys[0] {
		t.Fatalf("replaced key %v not found: %v", keys[0], res)
	}
	if !ix.Delete(keys[1]) || ix.Delete(keys[1]) || ix.Contains(keys[1]) {
		t.Fatalf("delete %v", keys[1])
	}
	if got := slices.Collect(ix.Keys()); len(got) != 2 {
		t.Fatalf("keys %v", got)
	}
	if err := ix.g.verify(); err != nil {
		t.Fatal(err)
	}
}

func TestEveryKeyKind(t *testing.T) {
	checkKeys(t, [3]string{"a", "", "c"})
	checkKeys(t, [3]namedString{"a", "b", ""})
	checkKeys(t, [3]int{0, -1, math.MaxInt})
	checkKeys(t, [3]namedInt{1, 0, math.MinInt})
	checkKeys(t, [3]int32{1, 2, math.MinInt32})
	checkKeys(t, [3]namedInt32{1, 2, 3})
	checkKeys(t, [3]int64{1, math.MaxInt64, 0})
	checkKeys(t, [3]namedInt64{1, 2, 3})
	checkKeys(t, [3]uint{1, 0, math.MaxUint})
	checkKeys(t, [3]namedUint{1, 2, 3})
	checkKeys(t, [3]uint32{0, math.MaxUint32, 7})
	checkKeys(t, [3]namedUint32{1, 2, 3})
	checkKeys(t, [3]uint64{math.MaxUint64, 0, 1})
	checkKeys(t, [3]namedUint64{1, 2, 3})
	checkKeys(t, [3][16]byte{{1}, {}, {15: 1}})
	checkKeys(t, [3]namedUUID{{1}, {2}, {3}})
}

func TestStringKeysAreCopied(t *testing.T) {
	ix, err := New[namedString](2, Options{})
	if err != nil {
		t.Fatal(err)
	}
	big := strings.Repeat("x", 1<<16)
	key := namedString(big[:3])
	if err := ix.Add(key, []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	stored := string(ix.g.keys[0])
	if stored != "xxx" || unsafe.StringData(stored) == unsafe.StringData(big) {
		t.Fatal("the stored key shares the caller's buffer")
	}
	// A replace must keep the owned key: a map assignment would overwrite byKey's copy.
	big2 := strings.Repeat("x", 1<<16)
	if err := ix.Add(namedString(big2[:3]), []float32{0, 1}); err != nil {
		t.Fatal(err)
	}
	for k := range ix.g.byKey {
		if unsafe.StringData(string(k)) == unsafe.StringData(big2) {
			t.Fatal("byKey shares the caller's buffer after a replace")
		}
	}
	if unsafe.StringData(string(ix.g.keys[1])) == unsafe.StringData(big2) {
		t.Fatal("the replacing node's key shares the caller's buffer")
	}
}

func TestStringKeyLengthLimit(t *testing.T) {
	defer func(old uint64) { maxKeyBytes = old }(maxKeyBytes)
	maxKeyBytes = 3
	ix, err := New[string](2, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.Add("abc", []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := ix.Add("abcd", []float32{1, 0}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("got %v, want ErrInvalidArgument", err)
	}
	if ix.Len() != 1 || ix.g.n != 1 {
		t.Fatal("a rejected key changed the index")
	}
}

func TestDeleteAndUpsert(t *testing.T) {
	const n = 500
	vecs := clustered(n+1, 8, 3)
	ix := build(t, vecs[:n], Options{M: 4})
	if err := ix.Add(7, slices.Clone(vecs[7])); err != nil { // same bits: a no-op
		t.Fatal(err)
	}
	if s := ix.Stats(); s.Nodes != n || s.Deleted != 0 || s.Keys != n {
		t.Fatalf("same vector made a node: %+v", s)
	}
	// Replace key 7 with a new vector: the old one must never be served for it.
	old := vecs[7]
	if err := ix.Add(7, vecs[n]); err != nil {
		t.Fatal(err)
	}
	if s := ix.Stats(); s.Nodes != n+1 || s.Deleted != 1 || s.Keys != n {
		t.Fatalf("after replace: %+v", s)
	}
	for _, q := range [][]float32{old, vecs[n]} {
		res, err := ix.Search(q, n, SearchOptions{Ef: n + 1})
		if err != nil || len(res) != n {
			t.Fatalf("%d results, %v", len(res), err)
		}
		for _, r := range res {
			want := vecs[r.Key]
			if r.Key == 7 {
				want = vecs[n]
			}
			if r.Distance != max(0, distance(q, want)) {
				t.Fatalf("key %d at %v, its vector is at %v", r.Key, r.Distance, distance(q, want))
			}
		}
	}
	for key := range 100 {
		if !ix.Delete(key) {
			t.Fatalf("Delete %d", key)
		}
	}
	if ix.Delete(0) || ix.Contains(0) || !ix.Contains(100) || ix.Len() != n-100 {
		t.Fatal("delete bookkeeping")
	}
	res, err := ix.Search(vecs[3], n, SearchOptions{Ef: n + 1})
	if err != nil || len(res) != n-100 {
		t.Fatalf("%d results, %v", len(res), err)
	}
	for _, r := range res {
		if r.Key < 100 {
			t.Fatalf("deleted key %d returned", r.Key)
		}
	}
	if err := ix.g.verify(); err != nil {
		t.Fatal(err)
	}
	if s := ix.Stats(); s.Dim != 8 || s.Keys != n-100 || s.Nodes != n+1 || s.Deleted != 101 || s.MaxLevel != int(ix.g.top) {
		t.Fatalf("stats %+v", s)
	}
}

func TestReplaceOnlyNode(t *testing.T) {
	ix, err := New[int](2, Options{})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range [][]float32{{1, 0}, {0, 1}, {-1, 0}} {
		if err := ix.Add(1, v); err != nil {
			t.Fatal(err)
		}
		res, err := ix.Search(v, 5, SearchOptions{})
		if err != nil || len(res) != 1 || res[0].Key != 1 || res[0].Distance != 0 {
			t.Fatalf("replace %d: %v, %v", i, res, err)
		}
	}
	if err := ix.g.verify(); err != nil {
		t.Fatal(err)
	}
}

// Replacing moves a key to a fresh node, so cover every pair of old and new levels.
// M=2 can leave a fresh node unreachable, so this checks the move itself.
func TestReplaceAcrossLevels(t *testing.T) {
	const n, d = 200, 8
	vecs := clustered(n+2000, d, 6)
	ix := build(t, vecs[:n], Options{M: 2, Seed: 4})
	g := ix.g
	seen := map[[2]uint8]bool{}
	for i, v := range vecs[n:] {
		key := i % n
		from := g.byKey[key]
		if err := ix.Add(key, v); err != nil {
			t.Fatal(err)
		}
		to := g.byKey[key]
		if to != g.n-1 || !g.isDead(from) || g.isDead(to) || !slices.Equal(g.vec(to), v) {
			t.Fatalf("replace %d: key moved from %d to %d of %d", i, from, to, g.n)
		}
		seen[[2]uint8{min(g.level[from], 2), min(g.level[to], 2)}] = true
	}
	if err := g.verify(); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 9 {
		t.Fatalf("covered level pairs %v, want all 9", slices.Collect(maps.Keys(seen)))
	}
}

func TestDeleteAllThenAdd(t *testing.T) {
	vecs := clustered(300, 8, 2)
	ix := build(t, vecs, Options{})
	for key := range vecs {
		ix.Delete(key)
	}
	if res, err := ix.Search(vecs[0], 3, SearchOptions{}); res != nil || err != nil || ix.Len() != 0 {
		t.Fatalf("empty: %v, %v", res, err)
	}
	if got := slices.Collect(ix.Keys()); len(got) != 0 {
		t.Fatalf("keys %v", got)
	}
	if err := ix.Add(1000, vecs[5]); err != nil {
		t.Fatal(err)
	}
	res, err := ix.Search(vecs[5], 3, SearchOptions{})
	if err != nil || len(res) != 1 || res[0].Key != 1000 {
		t.Fatalf("%v, %v", res, err)
	}
	if err := ix.g.verify(); err != nil {
		t.Fatal(err)
	}
}

func TestZeroIndexReadMethods(t *testing.T) {
	var ix Index[string]
	if ix.Delete("a") || ix.Contains("a") || ix.Len() != 0 || ix.Stats() != (Stats{}) {
		t.Fatal("zero Index")
	}
	for range ix.Keys() {
		t.Fatal("zero Index has keys")
	}
}

func TestKeysBreak(t *testing.T) {
	ix := build(t, clustered(10, 4, 1), Options{})
	n := 0
	for range ix.Keys() { // a Keys that kept yielding after break would panic here
		n++
		break
	}
	if n != 1 {
		t.Fatalf("%d keys before break", n)
	}
}

func TestKeysWhileMutating(t *testing.T) {
	vecs := clustered(400, 8, 5)
	ix := build(t, vecs[:200], Options{})
	seen := map[int]int{}
	for key := range ix.Keys() {
		seen[key]++
		if key%2 == 0 && key < 200 {
			ix.Delete(key) // the current key
		}
		if key < 100 {
			ix.Delete(key + 100) // a key not reached yet
			if err := ix.Add(key+200, vecs[key+200]); err != nil {
				t.Fatal(err)
			}
		}
	}
	for key := range 100 {
		if seen[key] != 1 {
			t.Fatalf("unchanged key %d seen %d times", key, seen[key])
		}
	}
	got := slices.Sorted(ix.Keys())
	want := []int{}
	for key := range 300 {
		if key%2 == 1 && key < 100 || key >= 200 {
			want = append(want, key)
		}
	}
	if !slices.Equal(got, want) {
		t.Fatalf("keys after mutation: %v", got)
	}
}

func TestBudgetStopsSearch(t *testing.T) {
	defer func(old int) { deadBudget = old }(deadBudget)
	deadBudget = 20
	vecs := clustered(2000, 8, 7)
	ix := build(t, vecs, Options{})
	for key := range 1990 {
		ix.Delete(key)
	}
	res, err := ix.Search(vecs[0], 5, SearchOptions{Ef: 50})
	if err != nil {
		t.Fatal(err)
	}
	if ix.Stats().BudgetStops != 1 {
		t.Fatalf("BudgetStops %d, want 1", ix.Stats().BudgetStops)
	}
	for _, r := range res {
		if r.Key < 1990 {
			t.Fatalf("deleted key %d returned", r.Key)
		}
	}
	deadBudget = 1 << 30
	if res, _ := ix.Search(vecs[0], 5, SearchOptions{Ef: 50}); len(res) != 5 || ix.Stats().BudgetStops != 1 {
		t.Fatalf("without the budget: %d results, %d stops", len(res), ix.Stats().BudgetStops)
	}
}

func TestBudgetCountsOnlyTombstones(t *testing.T) {
	defer func(old int) { deadBudget = old }(deadBudget)
	deadBudget = 5
	vecs := clustered(2000, 8, 7)
	ix := build(t, vecs, Options{})
	ix.Delete(1999)
	if _, err := ix.Search(vecs[0], 5, SearchOptions{Ef: 200}); err != nil {
		t.Fatal(err)
	}
	if s := ix.Stats().BudgetStops; s != 0 {
		t.Fatalf("live nodes counted against the budget: %d stops", s)
	}
}

// On a layer-0 chain from the entry whose first b nodes are tombstones, a budget of b
// lets the search through.
func TestBudgetIsExact(t *testing.T) {
	defer func(old int) { deadBudget = old }(deadBudget)
	for _, b := range []int{1, 2, 3} {
		deadBudget = b
		ix, err := New[int](2, Options{M: 2})
		if err != nil {
			t.Fatal(err)
		}
		for i := range 6 {
			if err := ix.Add(i, []float32{1, float32(i) * 0.01}); err != nil {
				t.Fatal(err)
			}
		}
		g := ix.g
		for id := range g.n {
			for layer := 0; layer <= int(g.level[id]); layer++ {
				clear(g.record(id, layer))
			}
		}
		chain := []uint32{g.entry}
		for id := range g.n {
			if id != g.entry {
				chain = append(chain, id)
			}
		}
		for i := 0; i+1 < len(chain); i++ {
			r := g.record(chain[i], 0)
			r[0], r[1] = 1, chain[i+1]
		}
		for _, id := range chain[:b] {
			ix.Delete(g.keys[id])
		}
		if err := g.verify(); err != nil {
			t.Fatal(err)
		}
		res, err := ix.Search(g.vec(chain[len(chain)-1]), 10, SearchOptions{Ef: 10})
		if err != nil {
			t.Fatal(err)
		}
		if ix.Stats().BudgetStops != 0 || len(res) != len(chain)-b {
			t.Fatalf("budget %d: %d results, %d stops", b, len(res), ix.Stats().BudgetStops)
		}
		deadBudget = b - 1 // one short of the tombstones on the path
		if _, err := ix.Search(g.vec(chain[len(chain)-1]), 10, SearchOptions{Ef: 10}); err != nil {
			t.Fatal(err)
		}
		if ix.Stats().BudgetStops != 1 {
			t.Fatalf("budget %d with %d tombstones on the path: %d stops, want 1", b-1, b, ix.Stats().BudgetStops)
		}
	}
}

// With 3 live keys the layer-0 beam is 3, so the search stops once W holds them instead
// of walking every tombstone.
func TestBeamIsBoundedByLiveKeys(t *testing.T) {
	defer func(old int) { deadBudget = old }(deadBudget)
	vecs := clustered(2000, 8, 7)
	ix := build(t, vecs, Options{})
	q := vecs[0]
	order := make([]int, len(vecs))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int { return cmp.Compare(distance(q, vecs[a]), distance(q, vecs[b])) })
	for _, key := range order[3:] {
		ix.Delete(key)
	}
	deadBudget = 1000
	res, err := ix.Search(q, 3, SearchOptions{Ef: 100})
	if err != nil {
		t.Fatal(err)
	}
	if ix.Stats().BudgetStops != 0 || len(res) != 3 {
		t.Fatalf("%d results, %d stops", len(res), ix.Stats().BudgetStops)
	}
}

// liveRecall measures recall@k over the keys still in the index.
func liveRecall(t *testing.T, ix *Index[int], vecs, queries [][]float32, k, ef int) (recall float64, short int) {
	t.Helper()
	var live [][]float32
	for key := range ix.Keys() {
		live = append(live, vecs[key])
	}
	truth := truthDistances(live, queries, k)
	hits := 0
	for i, q := range queries {
		res, err := ix.Search(q, k, SearchOptions{Ef: ef})
		if err != nil {
			t.Fatal(err)
		}
		if len(res) < k {
			short++
		}
		limit := max(0, truth[i][k-1]) + 2e-6
		for _, r := range res {
			if !ix.Contains(r.Key) {
				t.Fatalf("deleted key %d returned", r.Key)
			}
			if r.Distance <= limit {
				hits++
			}
		}
	}
	return float64(hits) / float64(k*len(queries)), short
}

// groups assigns each vector to the nearest of g random pivots, so deleting whole groups
// removes contiguous regions, as deleting a tenant or a document set does.
func groups(vecs [][]float32, g int, seed uint64) []int {
	s := splitmix(seed)
	pivots := make([][]float32, g)
	for i := range pivots {
		pivots[i] = vecs[s.next()%uint64(len(vecs))]
	}
	out := make([]int, len(vecs))
	for i, v := range vecs {
		best := float32(math.Inf(1))
		for p, pv := range pivots {
			if d := distance(v, pv); d < best {
				best, out[i] = d, p
			}
		}
	}
	return out
}

func TestChurn(t *testing.T) {
	skipUnderRace(t)
	const n, nq, d, k, cycles = 20000, 500, 128, 10, 5
	all := clustered(n+cycles*n*3/10+nq, d, 3)
	queries := all[len(all)-nq:]
	ix := build(t, all[:n], Options{M: 16, Seed: 3})
	s := splitmix(3)
	next := n
	for c := range cycles {
		keys := slices.Collect(ix.Keys())
		for range n * 3 / 10 {
			i := int(s.next() % uint64(len(keys)))
			ix.Delete(keys[i])
			keys[i] = keys[len(keys)-1]
			keys = keys[:len(keys)-1]
			if err := ix.Add(next, all[next]); err != nil {
				t.Fatal(err)
			}
			next++
		}
		if err := ix.g.verify(); err != nil {
			t.Fatalf("cycle %d: %v", c+1, err)
		}
	}
	if r, _ := liveRecall(t, ix, all, queries, k, 100); r < .999 {
		t.Fatalf("after %d cycles: recall %.4f, want >= .999", cycles, r)
	} else {
		t.Logf("after %d cycles: recall %.4f, %d of %d nodes deleted", cycles, r, ix.g.deadN, ix.g.n)
	}
	if u := ix.g.unreached(); u != 0 {
		t.Fatalf("%d live nodes unreached", u)
	}
}

// TestTombstones is a nightly scenario: 30% deleted uniformly or in whole groups.
func TestTombstones(t *testing.T) {
	if testing.Short() {
		t.Skip("nightly")
	}
	const n, nq, d, k = 100000, 1000, 128, 10
	all := clustered(n+nq, d, 7)
	base, queries := all[:n], all[n:]
	for _, mode := range []string{"uniform", "groups"} {
		ix := build(t, base, Options{M: 16, Seed: 7})
		if mode == "uniform" {
			s := splitmix(7)
			for ix.g.deadN < n*3/10 {
				ix.Delete(int(s.next() % n))
			}
		} else {
			grp := groups(base, 64, 7)
			for g := 0; ix.g.deadN < n*3/10; g++ {
				for key, kg := range grp {
					if kg == g {
						ix.Delete(key)
					}
				}
			}
		}
		if err := ix.g.verify(); err != nil {
			t.Fatal(err)
		}
		want := map[string]float64{"uniform": .999, "groups": .99}[mode]
		r, _ := liveRecall(t, ix, base, queries, k, 100)
		t.Logf("%s, %.1f%% deleted: recall@10 at ef=100 %.4f", mode, 100*float64(ix.g.deadN)/n, r)
		if r < want {
			t.Errorf("%s: recall %.4f at ef=100, want >= %v", mode, r, want)
		}
		if mode == "uniform" {
			r, short := liveRecall(t, ix, base, queries, k, 10)
			t.Logf("uniform: recall@10 at ef=10 %.4f, %d short results", r, short)
			if r < .90 || short != 0 {
				t.Errorf("uniform at ef=10: recall %.4f, %d short results", r, short)
			}
		}
	}
}

// fuzzVector reads a dim-4 vector with small integer coordinates, so ties, duplicates
// and zero vectors are common.
func fuzzVector(b []byte) []float32 {
	v := make([]float32, 4)
	for i := range v {
		v[i] = float32(int(b[i]%7) - 3)
	}
	return v
}

// FuzzOps runs byte-driven Add, Delete and Search sequences against a map model.
func FuzzOps(f *testing.F) {
	f.Add([]byte{0, 0, 1, 1, 2, 3, 4, 0, 2, 5, 6, 7, 8, 3, 1, 2, 3, 4, 5})
	f.Add([]byte{1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 0, 3, 0, 0, 0, 1, 9})
	f.Add([]byte{2, 0, 5, 1, 2, 3, 4, 1, 5, 4, 3, 2, 1, 2, 5, 0, 5, 9, 9, 9, 9, 3, 1, 2, 3, 4, 7})
	// M=2 with deletes and then searches, so the reach(landing) oracle runs on tombstones.
	f.Add([]byte("000000A010000000000200000000000000200000000000100700100"))
	f.Fuzz(func(t *testing.T, ops []byte) {
		if len(ops) == 0 {
			return
		}
		m := []int{2, 3, 16}[int(ops[0])%3]
		ops = ops[1:]
		ix, err := New[uint32](4, Options{M: m, EfConstruction: m + 4, Seed: 1})
		if err != nil {
			t.Fatal(err)
		}
		model := map[uint32][]float32{}
		stored := map[[4]float32]bool{} // every vector a key was served by: one node each
		var st searchState
		for len(ops) >= 6 {
			op, key, vec := ops[0]%4, uint32(ops[1]%32), fuzzVector(ops[2:6])
			ops = ops[6:]
			if op == 1 {
				vec[0] += 1e-6 // usually a near-duplicate of a vector op 0 adds
			}
			switch op {
			case 0, 1:
				q := make([]float32, 4)
				err := ix.Add(key, vec)
				if normalize(q, vec) != nil {
					if !errors.Is(err, ErrInvalidVector) {
						t.Fatalf("zero vector: %v", err)
					}
					continue
				}
				if err != nil {
					t.Fatal(err)
				}
				// The key is served by q or by a stored vector within mergeDist of it.
				served := slices.Clone(ix.g.vec(ix.g.byKey[key]))
				if !sameBits(served, q) && cosineDistance(served, q) > mergeDist {
					t.Fatalf("key %d added as %v is served by %v", key, q, served)
				}
				model[key] = served
				stored[[4]float32(served)] = true
			case 2:
				_, ok := model[key]
				if ix.Delete(key) != ok {
					t.Fatalf("Delete(%d) disagrees with the model", key)
				}
				delete(model, key)
			case 3:
				k := int(key%8) + 1
				q := make([]float32, 4)
				if normalize(q, vec) != nil {
					continue
				}
				stops := ix.Stats().BudgetStops
				res, err := ix.Search(vec, k, SearchOptions{Ef: maxBeam})
				if err != nil {
					t.Fatal(err)
				}
				if len(model) == 0 {
					if res != nil {
						t.Fatalf("empty index returned %v", res)
					}
					continue
				}
				_, landing, _ := ix.g.search(&st, q, k, maxBeam)
				seen := map[uint32]bool{}
				for i, r := range res {
					want, ok := model[r.Key]
					if !ok || seen[r.Key] {
						t.Fatalf("deleted or repeated key %d returned", r.Key)
					}
					seen[r.Key] = true
					if r.Distance != max(0, distance(q, want)) || i > 0 && r.Distance < res[i-1].Distance {
						t.Fatalf("result %d: %+v", i, r)
					}
				}
				// At full beam the search must find exactly the nearest live keys that
				// layer 0 connects to where the descent landed.
				var reachable []float32
				for _, id := range ix.g.reach(landing) {
					if !ix.g.isDead(id) {
						for range 1 + len(ix.g.alias[id]) {
							reachable = append(reachable, max(0, distance(q, ix.g.vec(id))))
						}
					}
				}
				slices.Sort(reachable)
				reachable = reachable[:min(k, len(reachable))]
				got := make([]float32, len(res))
				for i, r := range res {
					got[i] = r.Distance
				}
				if ix.Stats().BudgetStops == stops && !slices.Equal(got, reachable) {
					t.Fatalf("distances %v, reachable nearest %v", got, reachable)
				}
			}
			if err := ix.g.verify(); err != nil {
				t.Fatal(err)
			}
			if ix.Len() != len(model) || int(ix.g.n) != len(stored) {
				t.Fatalf("Len %d, model %d; %d nodes for %d vectors", ix.Len(), len(model), ix.g.n, len(stored))
			}
			if got, want := slices.Sorted(ix.Keys()), slices.Sorted(maps.Keys(model)); !slices.Equal(got, want) {
				t.Fatalf("keys %v, model %v", got, want)
			}
		}
	})
}
