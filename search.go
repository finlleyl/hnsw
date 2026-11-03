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

// Search returns up to k results, nearest first, 1 <= k <= 65536. It returns fewer than
// k if the index holds fewer keys or a node is unreachable. An empty index returns
// nil, nil. Tie order is unspecified. Errors: ErrDimension, ErrInvalidVector,
// ErrInvalidArgument.
func (ix *Index[K]) Search(query []float32, k int, o SearchOptions) ([]Result[K], error) {
	g := ix.g
	if g == nil {
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
	if len(query) != g.dim {
		return nil, fmt.Errorf("%w: dimension %d, want %d", ErrDimension, len(query), g.dim)
	}
	st := ix.pool.Get().(*searchState)
	defer ix.pool.Put(st)
	st.q = slices.Grow(st.q[:0], g.dim)[:g.dim]
	if err := normalize(st.q, query); err != nil {
		return nil, err
	}
	if g.live == 0 {
		return nil, nil
	}
	q := st.q
	ep := item{distance(q, g.vec(g.entry)), g.entry}
	for layer := int(g.top); layer > 0; layer-- {
		ep = g.searchLayer(st, q, ep, 1, layer)
	}
	// Once every live node is in W the beam is full and the normal stop rule applies.
	g.searchLayer(st, q, ep, min(max(ef, k), int(g.live)), 0)
	found := st.sortedW()
	found = found[:min(k, len(found))]
	out := make([]Result[K], len(found))
	for i, x := range found {
		out[i] = Result[K]{g.keys[x.id], max(0, x.dist)}
	}
	return out, nil
}

// searchLayer is Algorithm 2. It leaves the ef nearest nodes found in st.w and returns
// the nearest.
func (g *graph[K]) searchLayer(st *searchState, q []float32, ep item, ef, layer int) item {
	st.vis.reset(int(g.n))
	st.vis.visit(ep.id)
	st.cand = append(st.cand[:0], ep)
	st.w = append(st.w[:0], ep)
	nearest := ep
	for len(st.cand) > 0 {
		c := st.cand.pop()
		if len(st.w) >= ef && c.dist > st.w[0].dist {
			break
		}
		for _, e := range g.list(c.id, layer) {
			if !st.vis.visit(e) {
				continue
			}
			x := item{distance(q, g.vec(e)), e}
			if len(st.w) < ef || x.dist < st.w[0].dist {
				st.cand.push(x)
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
	return nearest
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
