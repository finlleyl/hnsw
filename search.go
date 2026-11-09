package hnsw

import (
	"fmt"
	"slices"
)

// SearchOptions tunes one Search. Zero fields take the index defaults.
type SearchOptions struct {
	_  [0]func() // non-comparable and keyed-only, so a later func field breaks nobody
	Ef int       // 0 uses Options.Ef; raised to k; maximum 65536
}

// Result is one search hit.
type Result[K Key] struct {
	Key      K
	Distance float32 // 1-cos clamped to >=0; an exact match may report a small positive value that grows with d
}

// searchState is scratch for one operation. Each operation resets what it uses first, so
// state left dirty by a panic cannot leak into the next one.
type searchState struct {
	vis    visited
	cand   minHeap
	w      maxHeap
	q      []float32 // normalized query or inserted vector
	sorted []item    // W in ascending order
	// Add only.
	sel, shrunk, links []item
	patches            []patch
	words              []uint32
}

func distance(a, b []float32) float32 { return 1 - dot(a, b) }

// deadBudget bounds the tombstones one search may expand, so a mostly deleted index
// cannot hold a search for long. It is a variable only so tests can lower it.
var deadBudget = 65536

// Search returns up to k results, nearest first, 1 <= k <= 65536. It returns fewer than
// k if the index holds fewer keys, if a key is unreachable, or if the tombstone walk
// budget ran out (Stats.BudgetStops counts this). An empty index returns nil, nil. Tie
// order is unspecified. Allocation is bounded by live keys, not by k. Errors:
// ErrDimension, ErrInvalidVector, ErrInvalidArgument.
func (ix *Index[K]) Search(query []float32, k int, o SearchOptions) ([]Result[K], error) {
	if ix.dim == 0 {
		return nil, fmt.Errorf("%w: zero Index", ErrInvalidArgument)
	}
	if k < 1 || k > maxBeam {
		return nil, fmt.Errorf("%w: k %d outside 1..%d", ErrInvalidArgument, k, maxBeam)
	}
	ef := o.Ef
	if ef == 0 {
		ef = ix.ef
	}
	if ef < 1 || ef > maxBeam {
		return nil, fmt.Errorf("%w: Ef %d outside 1..%d", ErrInvalidArgument, o.Ef, maxBeam)
	}
	if len(query) != ix.dim {
		return nil, fmt.Errorf("%w: dimension %d, want %d", ErrDimension, len(query), ix.dim)
	}
	st := ix.pool.Get().(*searchState)
	defer ix.pool.Put(st)
	st.q = slices.Grow(st.q[:0], ix.dim)[:ix.dim]
	if err := normalize(st.q, query); err != nil {
		return nil, err
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	g := ix.g
	if g.live == 0 {
		return nil, nil
	}
	found, _, stopped := g.search(st, st.q, k, ef)
	if stopped {
		ix.budgetStops.Add(1)
	}
	size := 0
	for _, x := range found {
		size += 1 + len(g.alias[x.id])
	}
	out := make([]Result[K], 0, min(size, k))
	for _, x := range found {
		d := max(0, x.dist)
		out = append(out, Result[K]{g.keys[x.id], d})
		for _, a := range g.alias[x.id] {
			if len(out) == k {
				break
			}
			out = append(out, Result[K]{a, d})
		}
		if len(out) == k {
			break
		}
	}
	return out, nil
}

// search is Algorithm 5 on a normalized query with at least one live key. It returns the
// k nearest live nodes found, the layer-0 node the descent landed on, and whether the
// tombstone budget stopped the search.
func (g *graph[K]) search(st *searchState, q []float32, k, ef int) (found []item, landing uint32, stopped bool) {
	ep := item{distance(q, g.vec(g.entry)), g.entry}
	for layer := int(g.top); layer > 0; layer-- {
		ep, _ = g.searchLayer(st, q, ep, 1, layer, false)
	}
	// Once every live node is in W the beam is full and the normal stop rule applies.
	_, stopped = g.searchLayer(st, q, ep, min(max(ef, k), int(g.live)), 0, true)
	found = st.sortedW()
	return found[:min(k, len(found))], ep.id, stopped
}

// searchLayer is Algorithm 2. It leaves the ef nearest nodes found in st.w and returns
// the nearest. Tombstones are ordinary nodes unless filter is set: then they are walked
// but never enter W, and expanding more than deadBudget of them stops the search.
func (g *graph[K]) searchLayer(st *searchState, q []float32, ep item, ef, layer int, filter bool) (nearest item, stopped bool) {
	st.vis.reset(int(g.n))
	st.vis.visit(ep.id)
	st.cand = append(st.cand[:0], ep)
	st.w = st.w[:0]
	if !filter || !g.isDead(ep.id) {
		st.w = append(st.w, ep)
	}
	nearest = ep
	deadPops := 0
	for len(st.cand) > 0 {
		c := st.cand.pop()
		if len(st.w) >= ef && c.dist > st.w[0].dist {
			break
		}
		if filter && g.isDead(c.id) {
			if deadPops++; deadPops > deadBudget {
				return nearest, true
			}
		}
		for _, e := range g.list(c.id, layer) {
			if !st.vis.visit(e) {
				continue
			}
			x := item{distance(q, g.vec(e)), e}
			if len(st.w) < ef || x.dist < st.w[0].dist {
				st.cand.push(x)
				if filter && g.isDead(e) {
					continue
				}
				st.w.push(x)
				if len(st.w) > ef {
					st.w.pop()
				}
				if x.less(nearest) {
					nearest = x
				}
			}
		}
	}
	return nearest, false
}

// sortedW empties W into st.sorted in (distance, ID) order.
func (st *searchState) sortedW() []item {
	n := len(st.w)
	st.sorted = slices.Grow(st.sorted[:0], n)[:n]
	for i := n - 1; i >= 0; i-- {
		st.sorted[i] = st.w.pop()
	}
	return st.sorted
}
