package hnsw

// selectNeighbors is Algorithm 4 without extendCandidates or keepPruned. cands are sorted
// in (distance, ID) order by distance to the base node. A candidate is kept unless a
// selected node is strictly closer to it than the base is. pending and q stand in for the
// node being inserted, whose vector is not stored yet. The result reuses out.
func (g *graph[K]) selectNeighbors(out, cands []item, m int, pending uint32, q []float32) []item {
	out = out[:0]
	for _, c := range cands {
		if len(out) >= m {
			break
		}
		cv := g.vecAt(c.id, pending, q)
		keep := true
		for _, r := range out {
			if distance(cv, g.vecAt(r.id, pending, q)) < c.dist {
				keep = false
				break
			}
		}
		if keep {
			out = append(out, c)
		}
	}
	return out
}
