package hnsw

import (
	"cmp"
	"fmt"
	"hash/maphash"
	"math"
	"slices"
	"testing"
)

func TestAliases(t *testing.T) {
	vecs := clustered(300, 8, 9)
	ix := build(t, vecs[:200], Options{M: 4})
	g := ix.g
	v := vecs[7]
	for _, key := range []int{1000, 1001, 1002} {
		if err := ix.Add(key, v); err != nil {
			t.Fatal(err)
		}
	}
	if s := ix.Stats(); s.Nodes != 200 || s.Keys != 203 || ix.Len() != 203 || len(g.alias[7]) != 3 {
		t.Fatalf("aliases made nodes: %+v, %v", s, g.alias)
	}
	for k := 1; k <= 6; k++ {
		res, err := ix.Search(v, k, SearchOptions{})
		if err != nil || len(res) != k {
			t.Fatalf("k=%d: %v, %v", k, res, err)
		}
		for i, r := range res[:min(k, 4)] { // the shared node's keys come first, each counting toward k
			if want := []int{7, 1000, 1001, 1002}[i]; r.Key != want || r.Distance != res[0].Distance {
				t.Fatalf("k=%d: result %d is %+v, want key %d", k, i, r, want)
			}
		}
	}
	if !slices.Equal(slices.Sorted(ix.Keys()), append(seq(200), 1000, 1001, 1002)) {
		t.Fatal("Keys misses aliases")
	}

	steps := []struct {
		do               func() bool
		primary, nodes   int
		aliases, deleted []int
	}{
		// Same key, same vector: nothing changes.
		{func() bool { return ix.Add(1001, v) == nil }, 7, 200, []int{1000, 1001, 1002}, nil},
		// The primary goes and the last alias takes its place.
		{func() bool { return ix.Delete(7) }, 1002, 200, []int{1000, 1001}, []int{7}},
		// A middle alias goes: swap-delete.
		{func() bool { return ix.Delete(1000) }, 1002, 200, []int{1001}, []int{1000}},
		// An alias moves to a new vector and gets a node of its own.
		{func() bool { return ix.Add(1001, vecs[250]) == nil }, 1002, 201, nil, nil},
		// The primary moves onto that vector, so the old node dies and the key becomes an alias.
		{func() bool { return ix.Add(1002, vecs[250]) == nil }, -1, 201, nil, nil},
		// The old vector comes back: its node revives under the new key.
		{func() bool { return ix.Add(1003, v) == nil }, 1003, 201, nil, nil},
		// A key moves from the revived node back to the alias node.
		{func() bool { return ix.Add(1003, vecs[250]) == nil }, -1, 201, nil, nil},
	}
	for i, s := range steps {
		if !s.do() {
			t.Fatalf("step %d failed", i)
		}
		if err := g.verify(); err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		if g.n != uint32(s.nodes) {
			t.Fatalf("step %d: %d nodes, want %d", i, g.n, s.nodes)
		}
		if s.primary < 0 {
			if !g.isDead(7) {
				t.Fatalf("step %d: node 7 holds key %d, want a tombstone", i, g.keys[7])
			}
		} else if g.isDead(7) || g.keys[7] != s.primary || !slices.Equal(g.alias[7], s.aliases) {
			t.Fatalf("step %d: node 7 holds %d and %v, want %d and %v", i, g.keys[7], g.alias[7], s.primary, s.aliases)
		}
		for _, key := range s.deleted {
			if ix.Contains(key) {
				t.Fatalf("step %d: deleted key %d remains", i, key)
			}
		}
	}
	if id := g.byKey[1001]; id != 200 || g.keys[200] != 1001 || !slices.Equal(g.alias[200], []int{1002, 1003}) {
		t.Fatalf("node 200 holds %d and %v", g.keys[200], g.alias[200])
	}
	res, err := ix.Search(vecs[250], 3, SearchOptions{})
	if err != nil || len(res) != 3 || res[0].Key != 1001 || res[1].Key != 1002 || res[2].Key != 1003 {
		t.Fatalf("%v, %v", res, err)
	}
}

func seq(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

// Keys copies each node's aliases before yielding, so deleting one mid-loop cannot make
// an unchanged key repeat or vanish.
func TestKeysWithAliasesWhileMutating(t *testing.T) {
	vecs := clustered(20, 4, 2)
	ix := build(t, vecs, Options{})
	for key := 100; key < 110; key++ {
		if err := ix.Add(key, vecs[3]); err != nil {
			t.Fatal(err)
		}
	}
	seen := map[int]int{}
	for key := range ix.Keys() {
		seen[key]++
		if key == 100 {
			ix.Delete(3)   // the primary, so the last alias is promoted
			ix.Delete(102) // a later alias, so the last one moves into its slot
		}
	}
	for key := range 20 {
		if seen[key] != 1 && key != 3 {
			t.Fatalf("key %d seen %d times", key, seen[key])
		}
	}
	for key := 100; key < 110; key++ {
		if seen[key] != 1 && key != 102 {
			t.Fatalf("key %d seen %d times", key, seen[key])
		}
	}
	if err := ix.g.verify(); err != nil {
		t.Fatal(err)
	}
}

func TestNegativeZeroMerges(t *testing.T) {
	ix, err := New[int](3, Options{})
	if err != nil {
		t.Fatal(err)
	}
	negZero := float32(math.Copysign(0, -1))
	for key, v := range [][]float32{{1, 0, 0}, {1, negZero, 0}, {2, 0, negZero}} {
		if err := ix.Add(key, v); err != nil {
			t.Fatal(err)
		}
	}
	if s := ix.Stats(); s.Nodes != 1 || s.Keys != 3 {
		t.Fatalf("%+v", s)
	}
}

// A vector within mergeDist of a stored one joins that node and is served by it.
func TestNearDuplicatesMerge(t *testing.T) {
	ix, err := New[int](3, Options{})
	if err != nil {
		t.Fatal(err)
	}
	g := ix.g
	add := func(key int, eps float32) {
		t.Helper()
		if err := ix.Add(key, []float32{1, eps, 0}); err != nil {
			t.Fatal(err)
		}
	}
	add(1, 0)
	add(2, 1e-3) // cosine distance 5e-7
	add(3, 2e-3) // 2e-6, too far
	if g.n != 2 || ix.Len() != 3 || g.byKey[2] != 0 || g.byKey[3] != 1 {
		t.Fatalf("%d nodes, keys at %v", g.n, g.byKey)
	}
	add(2, 1e-3) // replay: the key already sits on a node within mergeDist
	add(2, 5e-4)
	if g.n != 2 || g.byKey[2] != 0 || !slices.Equal(g.alias[0], []int{2}) {
		t.Fatalf("replay moved key 2: %v, %v", g.byKey, g.alias)
	}
	res, err := ix.Search([]float32{1, 1e-3, 0}, 2, SearchOptions{})
	if err != nil || len(res) != 2 || res[0].Key != 1 || res[1].Key != 2 || res[1].Distance != res[0].Distance {
		t.Fatalf("key 2 is not served by node 0's vector: %v, %v", res, err)
	}
	ix.Delete(1)
	ix.Delete(2)
	add(4, 1e-3) // a near-duplicate of a tombstone revives it
	if g.n != 2 || g.byKey[4] != 0 || g.isDead(0) || g.vec(0)[1] != 0 {
		t.Fatalf("%d nodes, key 4 at %d", g.n, g.byKey[4])
	}
	if err := g.verify(); err != nil {
		t.Fatal(err)
	}
}

// Replaying a key's Add keeps it on its node even when another node within mergeDist of
// the vector comes first in a search.
func TestNearDuplicateReplay(t *testing.T) {
	ix, err := New[int](3, Options{})
	if err != nil {
		t.Fatal(err)
	}
	at := func(a float64) []float32 { return []float32{float32(math.Cos(a)), float32(math.Sin(a)), 0} }
	// Cosine distances: key 2 is 7.2e-7 from key 1 and 6.1e-7 from key 3; 1 and 3 are 2.6e-6 apart.
	for key, a := range []float64{0, 1.2e-3, 2.3e-3} {
		if err := ix.Add(key+1, at(a)); err != nil {
			t.Fatal(err)
		}
	}
	if ix.g.n != 2 || ix.g.byKey[2] != 0 {
		t.Fatalf("%d nodes, key 2 at %d", ix.g.n, ix.g.byKey[2])
	}
	ix.Delete(1)
	if err := ix.Add(2, at(1.2e-3)); err != nil {
		t.Fatal(err)
	}
	if s := ix.Stats(); ix.g.byKey[2] != 0 || s.Deleted != 0 {
		t.Fatalf("replay moved key 2 to node %d: %+v", ix.g.byKey[2], s)
	}
}

// The float32 prefilter must not skip a node within mergeDist, also when the stored norm
// sits normTolerance below 1. Its dim*2^-23 term is a worst-case bound that random data
// does not reach.
func TestNearDuplicatePrefilter(t *testing.T) {
	s := splitmix(3)
	for _, d := range []int{3, 768} {
		built := 0
		for range 50 {
			v, e := make([]float32, d), make([]float32, d)
			for i := range v {
				v[i], e[i] = s.signed(), s.signed()
			}
			normalize(v, v)
			var ve float32
			for i := range v {
				ve += v[i] * e[i]
			}
			for i := range e { // orthogonal to v
				e[i] -= ve * v[i]
			}
			normalize(e, e)
			for _, c := range []struct {
				dist  float64
				nodes uint32
			}{{9e-7, 1}, {1.1e-6, 2}} {
				ix, err := New[int](d, Options{})
				if err != nil {
					t.Fatal(err)
				}
				u := make([]float32, d)
				for i := range u {
					u[i] = float32((1-c.dist)*float64(v[i]) + math.Sqrt(1-(1-c.dist)*(1-c.dist))*float64(e[i]))
				}
				short := make([]float32, d)
				f := math.Sqrt((1 - 8.5e-7) / squaredNorm(v)) // within normTolerance, so kept
				for i := range short {
					short[i] = float32(f * float64(v[i]))
				}
				if err := ix.Add(1, short); err != nil {
					t.Fatal(err)
				}
				if err := ix.Add(2, u); err != nil {
					t.Fatal(err)
				}
				sq := squaredNorm(ix.g.vec(0))
				got := cosineDistance(ix.g.vec(0), u)
				if math.Abs(got-c.dist) > 2e-8 || 1-sq < 7e-7 || 1-sq > normTolerance {
					continue // float32 rounding spoiled this draw
				}
				built++
				if ix.g.n != c.nodes {
					t.Fatalf("d=%d distance %g: %d nodes, want %d", d, got, ix.g.n, c.nodes)
				}
			}
		}
		if built < 60 {
			t.Fatalf("d=%d: only %d of 100 cases built as intended", d, built)
		}
	}
}

// Deleting and re-adding one vector revives its node instead of piling up tombstones.
func TestReAddLoopIsStable(t *testing.T) {
	vecs := clustered(1000, 16, 4)
	ix := build(t, vecs, Options{M: 8})
	for i := range 10000 {
		key := i % 50
		if !ix.Delete(key) {
			t.Fatalf("Delete %d", key)
		}
		if err := ix.Add(key, vecs[key]); err != nil {
			t.Fatal(err)
		}
		if s := ix.Stats(); s.Nodes != 1000 || s.Deleted != 0 {
			t.Fatalf("round %d: %+v", i, s)
		}
	}
	checkGraph(t, ix)
	for key := range 50 {
		res, err := ix.Search(vecs[key], 1, SearchOptions{Ef: 1000})
		if err != nil || len(res) != 1 || res[0].Key != key {
			t.Fatalf("key %d: %v, %v", key, res, err)
		}
	}
}

// A weak hash makes probe chains long; merging must still find exact matches only.
func TestWeakHash(t *testing.T) {
	defer func(old func(maphash.Seed, []float32) uint64) { hashVec = old }(hashVec)
	for name, h := range map[string]func(maphash.Seed, []float32) uint64{
		"constant": func(maphash.Seed, []float32) uint64 { return math.MaxUint64 - 3 }, // the probe wraps
		"low bits": func(_ maphash.Seed, v []float32) uint64 { return uint64(math.Float32bits(v[0]) & 3) },
	} {
		hashVec = h
		vecs := lattice(1500, 4, 7)
		ix := build(t, vecs, Options{M: 4})
		distinct := map[[4]float32]bool{}
		for _, v := range vecs {
			distinct[[4]float32(v)] = true
		}
		if int(ix.g.n) != len(distinct) || ix.Len() != len(vecs) {
			t.Fatalf("%s: %d nodes for %d vectors, %d keys", name, ix.g.n, len(distinct), ix.Len())
		}
		for key := range 300 {
			ix.Delete(key)
		}
		for key := range 300 {
			if err := ix.Add(key, vecs[key]); err != nil {
				t.Fatal(err)
			}
		}
		if int(ix.g.n) != len(distinct) || ix.g.deadN != 0 {
			t.Fatalf("%s: re-adding made nodes: %+v", name, ix.Stats())
		}
		if err := ix.g.verify(); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
}

// Deleting v and then filling its neighbourhood with 5000 points must not lose v once it
// is re-added. Inserts treat tombstones as ordinary nodes, so v keeps in-edges and the
// revive only clears its tombstone. The points are the nearest of more data, not
// isotropic noise around v, which makes a stored v a hub that Algorithm 4 strands
// thousands of nodes around.
func TestRevive(t *testing.T) {
	skipUnderRace(t)
	const n, near, d = 20000, 5000, 128
	all := clustered(n+60000, d, 1)
	ix := build(t, all[:n], Options{M: 16, Seed: 1})
	g := ix.g
	const key = 123
	v, id := all[key], g.byKey[key]
	ix.Delete(key)
	extra := all[n:]
	slices.SortFunc(extra, func(a, b []float32) int { return cmp.Compare(distance(v, a), distance(v, b)) })
	for i, u := range extra[:near] {
		if err := ix.Add(n+i, u); err != nil {
			t.Fatal(err)
		}
	}
	if err := ix.Add(key, v); err != nil {
		t.Fatal(err)
	}
	if g.byKey[key] != id || g.n != n+near {
		t.Fatalf("re-added at node %d of %d, want a revive of %d", g.byKey[key], g.n, id)
	}
	res, err := ix.Search(v, 1, SearchOptions{Ef: n + near})
	if err != nil || len(res) != 1 || res[0].Key != key {
		t.Fatalf("%v, %v", res, err)
	}
	checkGraph(t, ix)
}

// TestDuplicates inserts many keys sharing vectors, the case that turns literal
// Algorithm 4 into an absorbing set.
func TestDuplicates(t *testing.T) {
	skipUnderRace(t)
	const n, nq, d, k = 20000, 1000, 128, 10
	scenarios := []struct {
		name            string
		vectors, copies int
		shuffle         bool
	}{
		{"M+1 first", 1, 17, false},
		{"2M+1 first", 1, 33, false},
		{"1000 first", 1, 1000, false},
		{"5000 first", 1, 5000, false},
		{"5000 shuffled", 1, 5000, true},
		{"200x20 shuffled", 200, 20, true},
	}
	for _, sc := range scenarios {
		for _, seed := range []uint64{1, 2, 3} {
			t.Run(fmt.Sprintf("%s/seed=%d", sc.name, seed), func(t *testing.T) {
				t.Parallel()
				all := clustered(n+nq, d, seed)
				queries := all[n:]
				// Keys 0..vectors*copies-1 share vectors all[0..vectors-1]; the rest are distinct.
				vecs := make([][]float32, n)
				dups := sc.vectors * sc.copies
				for i := range vecs {
					vecs[i] = all[i]
					if i < dups {
						vecs[i] = all[i%sc.vectors]
					}
				}
				order := seq(n)
				if sc.shuffle {
					s := splitmix(seed)
					for i := n - 1; i > 0; i-- {
						j := int(s.next() % uint64(i+1))
						order[i], order[j] = order[j], order[i]
					}
				}
				ix, err := New[int](d, Options{M: 16, Seed: seed})
				if err != nil {
					t.Fatal(err)
				}
				for _, key := range order {
					if err := ix.Add(key, vecs[key]); err != nil {
						t.Fatal(err)
					}
				}
				if want := n - dups + sc.vectors; int(ix.g.n) != want {
					t.Fatalf("%d nodes, want %d", ix.g.n, want)
				}
				checkGraph(t, ix)
				if r := recall(t, ix, queries, truthDistances(vecs, queries, k), k, 100); r < .998 {
					t.Errorf("recall %.4f, want >= .998", r)
				}
				for c := range sc.vectors {
					res, err := ix.Search(all[c], 100, SearchOptions{})
					if err != nil {
						t.Fatal(err)
					}
					got := 0
					for _, r := range res {
						if r.Key < dups && r.Key%sc.vectors == c {
							got++
						}
					}
					if len(res) != 100 || got != min(sc.copies, 100) {
						t.Fatalf("vector %d: %d results, %d of its %d keys", c, len(res), got, sc.copies)
					}
				}
			})
		}
	}
}
