package hnsw

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
	top            uint8
	vecs           [][]float32 // chunks of chunkSize vectors; chunk 0 doubles until full
	adj0           []uint32    // n records of 2m+1 words: count, then links, unused slots zero
	up             []uint32    // upper records of m+1 words, appended in node-ID order
	upOff          []uint32    // index of a node's first upper record, valid when level > 0
	level          []uint8
	keys           []K
	byKey          map[K]uint32
}

func newGraph[K Key](dim, m int) *graph[K] {
	return &graph[K]{dim: dim, m: m, entry: noEntry, byKey: map[K]uint32{}}
}

func (g *graph[K]) vec(id uint32) []float32 {
	c := g.vecs[id>>chunkShift]
	i := int(id&(chunkSize-1)) * g.dim
	return c[i : i+g.dim : i+g.dim]
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

// publish stores a planned node: its vector from w.q, its key, and every list in
// w.patches, including the node's own.
func (g *graph[K]) publish(key K, w *searchState, id uint32, lv int) {
	g.appendVec(w.q)
	g.level = append(g.level, uint8(lv))
	g.upOff = append(g.upOff, uint32(g.records()))
	g.up = append(g.up, make([]uint32, lv*(g.m+1))...)
	g.adj0 = append(g.adj0, make([]uint32, 2*g.m+1)...)
	g.keys = append(g.keys, key)
	g.byKey[key] = id
	for _, p := range w.patches {
		r := g.record(p.id, p.layer)
		r[0] = uint32(copy(r[1:], w.words[p.off:p.off+p.n]))
		clear(r[1+p.n:])
	}
	if g.n == 0 || lv > int(g.top) {
		g.entry, g.top = id, uint8(lv)
	}
	g.n++
	g.live++
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
