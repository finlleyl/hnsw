package hnsw

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
)

// Errors returned by this package. Match them with errors.Is; they are wrapped with detail.
var (
	// ErrDimension means a vector's length differs from the index dimension.
	ErrDimension = errors.New("hnsw: wrong vector dimension")
	// ErrInvalidVector means a vector holds NaN or Inf, or is the zero vector.
	ErrInvalidVector = errors.New("hnsw: invalid vector")
	// ErrInvalidArgument means an option or argument is out of range or nil.
	ErrInvalidArgument = errors.New("hnsw: invalid argument")
	// ErrFull means the index ran out of node IDs or upper-layer records.
	ErrFull = errors.New("hnsw: index full")
	// ErrCorrupt means a file is truncated or fails validation.
	ErrCorrupt = errors.New("hnsw: corrupt file")
	// ErrVersion means a file has a format version this package cannot read.
	ErrVersion = errors.New("hnsw: unsupported file version")
	// ErrKeyType means a file's keys do not fit the requested key type.
	ErrKeyType = errors.New("hnsw: key type mismatch")
)

// Key lists the persistable key types. A type outside this set fails at compile time,
// never at save time. ~[16]byte admits UUID types.
type Key interface {
	~string | ~int | ~int32 | ~int64 | ~uint | ~uint32 | ~uint64 | ~[16]byte
}

// Options configures New. Zero fields take their defaults.
type Options struct {
	M              int    // links per node on upper layers, 2M on layer 0. 0 means 16. Range 2..512.
	EfConstruction int    // build beam. 0 means max(200, M). Range M..65536.
	Ef             int    // default search beam. 0 means 100. Range 1..65536.
	Seed           uint64 // level draws; the same seed, GOARCH and insert order give the same graph
}

const (
	maxDim  = 65536
	maxBeam = 65536 // cap on k, Ef and EfConstruction
)

// Index is an approximate nearest neighbour index. Searches may run concurrently with
// each other, but Add must not run concurrently with any other method.
type Index[K Key] struct {
	g    *graph[K]
	pool sync.Pool   // *searchState for Search
	w    searchState // writer scratch for Add
	efc  int         // immutable after New, like the fields below
	ef   int
	ml   float64 // level multiplier 1/ln(M)
	seed uint64
}

// New validates everything up front. dim is fixed for the life of the index, 1..65536.
// An Index must come from New. On a zero Index, methods that return an error return
// ErrInvalidArgument and the others return zero values.
func New[K Key](dim int, o Options) (*Index[K], error) {
	if dim < 1 || dim > maxDim {
		return nil, fmt.Errorf("%w: dimension %d outside 1..%d", ErrInvalidArgument, dim, maxDim)
	}
	if o.M == 0 {
		o.M = 16
	}
	if o.M < 2 || o.M > 512 {
		return nil, fmt.Errorf("%w: M %d outside 2..512", ErrInvalidArgument, o.M)
	}
	if o.EfConstruction == 0 {
		o.EfConstruction = max(200, o.M)
	}
	if o.EfConstruction < o.M || o.EfConstruction > maxBeam {
		return nil, fmt.Errorf("%w: EfConstruction %d outside %d..%d", ErrInvalidArgument, o.EfConstruction, o.M, maxBeam)
	}
	if o.Ef == 0 {
		o.Ef = 100
	}
	if o.Ef < 1 || o.Ef > maxBeam {
		return nil, fmt.Errorf("%w: Ef %d outside 1..%d", ErrInvalidArgument, o.Ef, maxBeam)
	}
	ix := &Index[K]{
		g:    newGraph[K](dim, o.M),
		efc:  o.EfConstruction,
		ef:   o.Ef,
		ml:   1 / math.Log(float64(o.M)),
		seed: o.Seed,
	}
	ix.pool.New = func() any { return new(searchState) }
	return ix, nil
}

// Add inserts key with a copy of vector, validated and normalized. On error the index is
// unchanged. Errors: ErrDimension, ErrInvalidVector (NaN, Inf, zero), ErrFull,
// ErrInvalidArgument (zero Index, key already present).
func (ix *Index[K]) Add(key K, vector []float32) error {
	g := ix.g
	if g == nil {
		return fmt.Errorf("%w: zero Index", ErrInvalidArgument)
	}
	if len(vector) != g.dim {
		return fmt.Errorf("%w: dimension %d, want %d", ErrDimension, len(vector), g.dim)
	}
	w := &ix.w
	w.q = slices.Grow(w.q[:0], g.dim)[:g.dim]
	if err := normalize(w.q, vector); err != nil {
		return err
	}
	// shortcut: an existing key is rejected, not replaced; upsert replaces this check.
	if _, ok := g.byKey[key]; ok {
		return fmt.Errorf("%w: key already present", ErrInvalidArgument)
	}
	if g.n >= maxNodes {
		return fmt.Errorf("%w: %d nodes", ErrFull, g.n)
	}
	id := g.n
	lv := ix.levelOf(id)
	if int64(g.records())+int64(lv) > maxRecords {
		return fmt.Errorf("%w: upper-layer records", ErrFull)
	}
	ix.plan(id, lv)
	g.publish(key, w, id, lv)
	return nil
}

// levelOf draws a node's top layer, min(floor(-ln(u)*mL), 31). u comes from a hash of the
// ID, so no generator state needs saving; this relies on IDs never being reused.
func (ix *Index[K]) levelOf(id uint32) int {
	u := float64(splitmix64(ix.seed+uint64(id))>>11+1) * 0x1p-53 // (0, 1]
	return min(int(-math.Log(u)*ix.ml), maxLevel)
}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// plan computes the new node's lists and its neighbours' updated lists into ix.w without
// writing the graph. Each layer's search reads only that layer's lists, and the updates
// for a layer touch only that layer, so deferring them gives the graph that in-place
// insertion in the paper's order would.
func (ix *Index[K]) plan(id uint32, lv int) {
	g, w := ix.g, &ix.w
	w.patches, w.words = w.patches[:0], w.words[:0]
	if g.n == 0 {
		return
	}
	q := w.q
	ep := item{distance(q, g.vec(g.entry)), g.entry}
	for layer := int(g.top); layer > lv; layer-- {
		ep = g.searchLayer(w, q, ep, 1, layer)
	}
	for layer := min(int(g.top), lv); layer >= 0; layer-- {
		g.searchLayer(w, q, ep, ix.efc, layer)
		cands := w.sortedW()
		ep = cands[0]
		w.sel = g.selectNeighbors(w.sel, cands, g.m, id, q)
		off := len(w.words)
		for _, s := range w.sel {
			w.words = append(w.words, s.id)
		}
		w.patches = append(w.patches, patch{id, layer, off, len(w.sel)})
		for _, s := range w.sel {
			ix.planBackLink(s.id, id, layer)
		}
	}
}

// planBackLink records e's list on layer with id added, re-selecting with Algorithm 4
// when the list is already at the layer's cap.
func (ix *Index[K]) planBackLink(e, id uint32, layer int) {
	g, w := ix.g, &ix.w
	old := g.list(e, layer)
	off := len(w.words)
	if len(old) < g.degree(layer) {
		w.words = append(append(w.words, old...), id)
		w.patches = append(w.patches, patch{e, layer, off, len(old) + 1})
		return
	}
	ev := g.vec(e)
	w.links = w.links[:0]
	for _, x := range old {
		w.links = append(w.links, item{distance(ev, g.vec(x)), x})
	}
	w.links = append(w.links, item{distance(ev, w.q), id})
	slices.SortFunc(w.links, compareItems)
	w.shrunk = g.selectNeighbors(w.shrunk, w.links, g.degree(layer), id, w.q)
	for _, s := range w.shrunk {
		w.words = append(w.words, s.id)
	}
	w.patches = append(w.patches, patch{e, layer, off, len(w.shrunk)})
}

// Len returns the number of keys in the index.
func (ix *Index[K]) Len() int {
	if ix.g == nil {
		return 0
	}
	return int(ix.g.live)
}
