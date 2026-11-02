package hnsw

import "slices"

// visited is a set of node IDs. Clearing costs only the words the last operation touched.
// Words outside touched are always zero, spare capacity included.
type visited struct {
	bits    []uint64
	touched []uint32 // indices of non-zero words in bits
}

// reset empties the set and sizes it for IDs below n. Callers run it at the start of an
// operation, not the end, so a panic that left the set dirty cannot lower the next
// operation's quality.
func (v *visited) reset(n int) {
	for _, w := range v.touched {
		v.bits[w] = 0
	}
	words := (n + 63) / 64
	v.bits = slices.Grow(v.bits[:0], words)[:words]
	v.touched = slices.Grow(v.touched[:0], words)
}

// visit adds id and reports whether it was absent. id must be below reset's n.
func (v *visited) visit(id uint32) bool {
	w, mask := id/64, uint64(1)<<(id%64)
	old := v.bits[w]
	if old&mask != 0 {
		return false
	}
	if old == 0 {
		v.touched = append(v.touched, w)
	}
	v.bits[w] = old | mask
	return true
}
