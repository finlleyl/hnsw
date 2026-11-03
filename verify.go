package hnsw

import (
	"fmt"
	"math"
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
	return nil
}

// unreached counts nodes that a walk over layer-0 links from the entry misses. It is a
// quality measure, not an invariant: directed edges do not guarantee reachability.
func (g *graph[K]) unreached() int {
	if g.n == 0 {
		return 0
	}
	var seen visited
	seen.reset(int(g.n))
	seen.visit(g.entry)
	stack := []uint32{g.entry}
	reached := 1
	for len(stack) > 0 {
		id := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, e := range g.list(id, 0) {
			if seen.visit(e) {
				reached++
				stack = append(stack, e)
			}
		}
	}
	return int(g.n) - reached
}
