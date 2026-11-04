package hnsw

import (
	"errors"
	"fmt"
	"iter"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
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

// maxKeyBytes is the longest string key, because the file format stores a u32 length.
// It is a variable only so tests can lower it.
var maxKeyBytes uint64 = 1<<32 - 1

// Index is an approximate nearest neighbour index. Searches, Contains, Len, Keys and
// Stats may run concurrently with each other, but Add and Delete must not run
// concurrently with any other method.
type Index[K Key] struct {
	g           *graph[K]
	pool        sync.Pool   // *searchState for Search
	w           searchState // writer scratch for Add
	budgetStops atomic.Uint64
	efc         int // immutable after New, like the fields below
	ef          int
	ml          float64 // level multiplier 1/ln(M)
	seed        uint64
	stringKey   bool // K's underlying type is string
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
		// string(k) does not compile for a union with ~[16]byte, and unsafe is not used.
		stringKey: reflect.TypeFor[K]().Kind() == reflect.String,
	}
	ix.pool.New = func() any { return new(searchState) }
	return ix, nil
}

// Add inserts or replaces key. The vector is validated, copied and normalized. Same key
// and same vector is a no-op, so replay is idempotent. On error the index is unchanged.
// Errors: ErrDimension, ErrInvalidVector (NaN, Inf, zero), ErrFull, ErrInvalidArgument
// (zero Index, string key over 2^32-1 bytes).
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
	old, exists := g.byKey[key]
	if exists {
		// Normalized vectors hold no NaN or -0, so == is bit equality here.
		if slices.Equal(g.vec(old), w.q) {
			return nil
		}
		key = g.keys[old] // already owned
	} else {
		var err error
		if key, err = ix.ownKey(key); err != nil {
			return err
		}
		old = noEntry
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
	g.publish(key, w, id, lv, old)
	return nil
}

// ownKey copies a string key, so a substring cannot pin the caller's larger buffer, and
// rejects one too long to save.
func (ix *Index[K]) ownKey(key K) (K, error) {
	if !ix.stringKey {
		return key, nil
	}
	v := reflect.ValueOf(&key).Elem()
	if uint64(v.Len()) > maxKeyBytes {
		return key, fmt.Errorf("%w: string key of %d bytes", ErrInvalidArgument, v.Len())
	}
	v.SetString(strings.Clone(v.String()))
	return key, nil
}

// Delete removes key and reports whether it existed. The node stays in the graph as a
// tombstone that searches walk through, so its embedding stays in memory.
func (ix *Index[K]) Delete(key K) bool {
	g := ix.g
	if g == nil {
		return false
	}
	id, ok := g.byKey[key]
	if !ok {
		return false
	}
	delete(g.byKey, key)
	g.kill(id)
	return true
}

// Contains reports whether key is in the index.
func (ix *Index[K]) Contains(key K) bool {
	if ix.g == nil {
		return false
	}
	_, ok := ix.g.byKey[key]
	return ok
}

// Keys iterates over the keys. The loop body may call any method. A key not changed
// during iteration appears exactly once; keys added, deleted or replaced meanwhile may
// appear, repeat or be missed.
func (ix *Index[K]) Keys() iter.Seq[K] {
	return func(yield func(K) bool) {
		g := ix.g
		if g == nil {
			return
		}
		for id := uint32(0); id < g.n; id++ { // g.n is reread, since the body may Add
			if !g.isDead(id) && !yield(g.keys[id]) {
				return
			}
		}
	}
}

// Stats describes the index. Fields will only be added.
type Stats struct {
	Dim, Keys, Nodes, Deleted, MaxLevel int    // Deleted counts tombstoned nodes
	BudgetStops                         uint64 // searches cut short by the tombstone walk budget
}

func (ix *Index[K]) Stats() Stats {
	g := ix.g
	if g == nil {
		return Stats{}
	}
	return Stats{
		Dim:         g.dim,
		Keys:        int(g.live),
		Nodes:       int(g.n),
		Deleted:     int(g.deadN),
		MaxLevel:    int(g.top),
		BudgetStops: ix.budgetStops.Load(),
	}
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
		ep, _ = g.searchLayer(w, q, ep, 1, layer, false)
	}
	for layer := min(int(g.top), lv); layer >= 0; layer-- {
		g.searchLayer(w, q, ep, ix.efc, layer, false)
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
