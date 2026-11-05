package hnsw

import (
	"encoding/binary"
	"hash/maphash"
	"math"
	"slices"
)

const (
	chunkShift = 12 // 4096 vectors per chunk, so growth never copies more than one chunk
	chunkSize  = 1 << chunkShift
	maxLevel   = 31
	maxNodes   = 1<<32 - 2
	maxRecords = 1<<32 - 1 // upper-layer records, the range of upOff
	noEntry    = ^uint32(0)
)

// graph holds the index data. Every edge read goes through list or record.
type graph[K Key] struct {
	dim, m         int // immutable; layer 0 holds up to 2m links
	n, live, entry uint32
	deadN          uint32 // tombstoned nodes, so live == n - deadN
	top            uint8
	vecs           [][]float32 // chunks of chunkSize vectors; chunk 0 doubles until full
	adj0           []uint32    // n records of 2m+1 words: count, then links, unused slots zero
	up             []uint32    // upper records of m+1 words, appended in node-ID order
	upOff          []uint32    // index of a node's first upper record, valid when level > 0
	level          []uint8
	dead           []uint64       // tombstone bitset
	keys           []K            // primary key per node; zero value when dead
	alias          map[uint32][]K // extra keys of live nodes; empty for almost every index
	byKey          map[K]uint32
	byVec          map[uint64]uint32 // vector hash to node, probing upward on collision
	seed           maphash.Seed      // for byVec, which is never saved
}

func newGraph[K Key](dim, m int) *graph[K] {
	return &graph[K]{
		dim: dim, m: m, entry: noEntry,
		alias: map[uint32][]K{}, byKey: map[K]uint32{}, byVec: map[uint64]uint32{},
		seed: maphash.MakeSeed(),
	}
}

// hashVec hashes a vector's bits. It is a variable only so tests can weaken it.
var hashVec = func(seed maphash.Seed, v []float32) uint64 {
	var h maphash.Hash
	h.SetSeed(seed)
	var b [256]byte
	for len(v) > 0 {
		n := min(len(v), len(b)/4)
		for i, x := range v[:n] {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
		}
		h.Write(b[:4*n])
		v = v[n:]
	}
	return h.Sum64()
}

// findVec returns the node holding v's exact bits, live or dead, or noEntry and the
// free byVec slot where a new node for v belongs. byVec never loses entries before a
// rebuild, so every probe chain is unbroken.
func (g *graph[K]) findVec(v []float32) (id uint32, slot uint64) {
	h := hashVec(g.seed, v)
	for {
		id, ok := g.byVec[h]
		if !ok {
			return noEntry, h
		}
		// Stored vectors hold no NaN or -0, so == is bit equality here.
		if slices.Equal(g.vec(id), v) {
			return id, h
		}
		h++
	}
}

func (g *graph[K]) vec(id uint32) []float32 {
	c := g.vecs[id>>chunkShift]
	i := int(id&(chunkSize-1)) * g.dim
	return c[i : i+g.dim : i+g.dim]
}

func (g *graph[K]) isDead(id uint32) bool { return g.dead[id/64]&(1<<(id%64)) != 0 }

// detach removes key from live node id. The last alias replaces a departing primary, and
// a node left with no key is tombstoned. byKey is the caller's to update.
func (g *graph[K]) detach(id uint32, key K) {
	as := g.alias[id]
	if g.keys[id] == key {
		if len(as) == 0 {
			g.kill(id)
			return
		}
		g.keys[id] = as[len(as)-1]
	} else {
		as[slices.Index(as, key)] = as[len(as)-1]
	}
	var zero K
	as[len(as)-1] = zero // so a removed string key is not pinned
	if len(as) == 1 {
		delete(g.alias, id)
	} else {
		g.alias[id] = as[:len(as)-1]
	}
}

// kill tombstones a live node with no aliases. Its links stay, so no edge can dangle and
// searches still walk through it. byKey is the caller's to update.
func (g *graph[K]) kill(id uint32) {
	g.dead[id/64] |= 1 << (id % 64)
	var zero K
	g.keys[id] = zero
	g.live--
	g.deadN++
}

// vecAt is vec, except that pending, a node being inserted, maps to its unstored vector q.
func (g *graph[K]) vecAt(id, pending uint32, q []float32) []float32 {
	if id == pending {
		return q
	}
	return g.vec(id)
}

// degree is the link cap of a layer.
func (g *graph[K]) degree(layer int) int {
	if layer == 0 {
		return 2 * g.m
	}
	return g.m
}

// record returns id's full record on layer: the count, then every slot.
func (g *graph[K]) record(id uint32, layer int) []uint32 {
	s := g.degree(layer) + 1
	o := int(id) * s
	if layer > 0 {
		o = (int(g.upOff[id]) + layer - 1) * s
		return g.up[o : o+s : o+s]
	}
	return g.adj0[o : o+s : o+s]
}

// list returns id's links on layer.
func (g *graph[K]) list(id uint32, layer int) []uint32 {
	r := g.record(id, layer)
	return r[1 : 1+r[0]]
}

// records counts upper-layer records.
func (g *graph[K]) records() int {
	return len(g.up) / (g.m + 1)
}

// patch is a planned list: id's links on layer are words[off : off+n] of the plan.
type patch struct {
	id     uint32
	layer  int
	off, n int
}

// publish applies a planned Add of key to node id: a new node when id is g.n, with its
// vector from w.q, level lv and byVec slot; a revived tombstone; or an alias of a live
// node. Every list in w.patches is stored. The key's previous node, old, loses the key
// in the same step; noEntry means none.
func (g *graph[K]) publish(key K, w *searchState, id uint32, lv int, slot uint64, old uint32) {
	switch {
	case id == g.n:
		g.appendVec(w.q)
		g.level = append(g.level, uint8(lv))
		g.upOff = append(g.upOff, uint32(g.records()))
		g.up = append(g.up, make([]uint32, lv*(g.m+1))...)
		g.adj0 = append(g.adj0, make([]uint32, 2*g.m+1)...)
		if id%64 == 0 {
			g.dead = append(g.dead, 0)
		}
		g.keys = append(g.keys, key)
		g.byVec[slot] = id
		if g.n == 0 || lv > int(g.top) {
			g.entry, g.top = id, uint8(lv)
		}
		g.n++
		g.live++
	case g.isDead(id):
		g.dead[id/64] &^= 1 << (id % 64)
		g.keys[id] = key
		g.live++
		g.deadN--
	default:
		g.alias[id] = append(g.alias[id], key)
	}
	g.byKey[key] = id
	for _, p := range w.patches {
		r := g.record(p.id, p.layer)
		r[0] = uint32(copy(r[1:], w.words[p.off:p.off+p.n]))
		clear(r[1+p.n:])
	}
	if old != noEntry {
		g.detach(old, key)
	}
}

func (g *graph[K]) appendVec(v []float32) {
	c := int(g.n >> chunkShift)
	if c == len(g.vecs) {
		size := chunkSize * g.dim
		if c == 0 {
			size = g.dim
		}
		g.vecs = append(g.vecs, make([]float32, 0, size))
	}
	if s := g.vecs[c]; len(s) == cap(s) { // only chunk 0 fills before its last vector
		grown := make([]float32, len(s), min(2*cap(s), chunkSize*g.dim))
		copy(grown, s)
		g.vecs[c] = grown
	}
	g.vecs[c] = append(g.vecs[c], v...)
}
