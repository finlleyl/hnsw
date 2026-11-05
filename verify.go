package hnsw

import (
	"fmt"
	"math"
	"math/bits"
)

// verify checks the graph invariants and returns the first violation, wrapping ErrCorrupt.
// It checks the layout first, so the checks after it cannot index out of range.
func (g *graph[K]) verify() error {
	bad := func(format string, a ...any) error {
		return fmt.Errorf("%w: %s", ErrCorrupt, fmt.Sprintf(format, a...))
	}
	n := int(g.n)
	if g.n > maxNodes {
		return bad("%d nodes", g.n)
	}
	if len(g.level) != n || len(g.upOff) != n || len(g.keys) != n || len(g.adj0) != n*(2*g.m+1) {
		return bad("array lengths do not match %d nodes", n)
	}
	if len(g.dead) != (n+63)/64 {
		return bad("%d tombstone words for %d nodes", len(g.dead), n)
	}
	if len(g.vecs) != (n+chunkSize-1)/chunkSize {
		return bad("%d vector chunks for %d nodes", len(g.vecs), n)
	}
	for c, s := range g.vecs {
		if want := min(n-c*chunkSize, chunkSize) * g.dim; len(s) != want {
			return bad("vector chunk %d holds %d values, want %d", c, len(s), want)
		}
	}
	// I4: entry and levels.
	if (n == 0) != (g.entry == noEntry) || (n == 0 && g.top != 0) {
		return bad("entry %d, top %d with %d nodes", g.entry, g.top, n)
	}
	if g.top > maxLevel || (n > 0 && (g.entry >= g.n || g.level[g.entry] != g.top)) {
		return bad("entry %d is not on top layer %d", g.entry, g.top)
	}
	records := 0
	for id, lv := range g.level {
		if lv > g.top {
			return bad("node %d level %d above top %d", id, lv, g.top)
		}
		if lv > 0 && int(g.upOff[id]) != records {
			return bad("node %d upper offset %d, want %d", id, g.upOff[id], records)
		}
		records += int(lv)
	}
	if int64(records) > maxRecords || len(g.up) != records*(g.m+1) {
		return bad("%d upper words for %d records", len(g.up), records)
	}
	// I9: counts and tombstones.
	deadN := 0
	for i, w := range g.dead {
		if i == len(g.dead)-1 && n%64 != 0 && w>>(n%64) != 0 {
			return bad("tombstone bits set past node %d", n)
		}
		deadN += bits.OnesCount64(w)
	}
	if deadN != int(g.deadN) || g.live != g.n-g.deadN {
		return bad("live %d, deleted %d, but %d of %d nodes are tombstones", g.live, g.deadN, deadN, n)
	}
	// I5 and I6: byKey maps exactly the primaries of live nodes and their aliases, and
	// tombstones hold no key.
	for id, as := range g.alias {
		if id >= g.n || g.isDead(id) || len(as) == 0 {
			return bad("alias list of %d keys on node %d", len(as), id)
		}
	}
	var zero K
	held := make(map[K]bool, len(g.byKey))
	hold := func(id uint32, key K) error {
		if got, ok := g.byKey[key]; !ok || got != id || held[key] {
			return bad("node %d key maps to %d or repeats", id, got)
		}
		held[key] = true
		return nil
	}
	for id := range g.n {
		if g.isDead(id) {
			if g.keys[id] != zero {
				return bad("tombstone %d holds a key", id)
			}
			continue
		}
		if err := hold(id, g.keys[id]); err != nil {
			return err
		}
		for _, a := range g.alias[id] {
			if err := hold(id, a); err != nil {
				return err
			}
		}
	}
	if len(held) != len(g.byKey) {
		return bad("%d keys held, %d in the key map", len(held), len(g.byKey))
	}
	// I1-I3: links.
	var seen visited
	for id := range g.n {
		for layer := 0; layer <= int(g.level[id]); layer++ {
			r := g.record(id, layer)
			if int(r[0]) > g.degree(layer) {
				return bad("node %d layer %d degree %d over cap %d", id, layer, r[0], g.degree(layer))
			}
			for _, x := range r[1+r[0]:] {
				if x != 0 {
					return bad("node %d layer %d unused slot not zero", id, layer)
				}
			}
			seen.reset(n)
			for _, e := range r[1 : 1+r[0]] {
				switch {
				case e >= g.n:
					return bad("node %d layer %d links to %d of %d nodes", id, layer, e, n)
				case int(g.level[e]) < layer:
					return bad("node %d layer %d links to %d at level %d", id, layer, e, g.level[e])
				case e == id:
					return bad("node %d layer %d links to itself", id, layer)
				case !seen.visit(e):
					return bad("node %d layer %d links to %d twice", id, layer, e)
				}
			}
		}
	}
	// I8: vectors are finite, unit length and free of -0, which is exactly what makes
	// normalize return them unchanged.
	buf := make([]float32, g.dim)
	for id := range g.n {
		v := g.vec(id)
		if normalize(buf, v) != nil {
			return bad("node %d vector is not finite and nonzero", id)
		}
		for i := range v {
			if math.Float32bits(v[i]) != math.Float32bits(buf[i]) {
				return bad("node %d vector is not normalized", id)
			}
		}
	}
	// I7: byVec indexes every node, and the probe for each node's bits stops at that node,
	// so no two nodes share bits.
	if len(g.byVec) != n {
		return bad("%d vector hashes for %d nodes", len(g.byVec), n)
	}
	for _, id := range g.byVec {
		if id >= g.n {
			return bad("vector hash maps to node %d of %d", id, n)
		}
	}
	for id := range g.n {
		if got, _ := g.findVec(g.vec(id)); got != id {
			return bad("node %d vector found at node %d", id, got)
		}
	}
	return nil
}

// unreached counts live nodes that a walk over layer-0 links from the entry misses. It is
// a quality measure, not an invariant: directed edges do not guarantee reachability.
func (g *graph[K]) unreached() int {
	if g.n == 0 {
		return 0
	}
	missed := int(g.live)
	for _, id := range g.reach(g.entry) {
		if !g.isDead(id) {
			missed--
		}
	}
	return missed
}

// reach returns the nodes a walk over layer-0 links from id visits, tombstones included.
func (g *graph[K]) reach(id uint32) []uint32 {
	var seen visited
	seen.reset(int(g.n))
	seen.visit(id)
	out := []uint32{id}
	for i := 0; i < len(out); i++ {
		for _, e := range g.list(out[i], 0) {
			if seen.visit(e) {
				out = append(out, e)
			}
		}
	}
	return out
}
