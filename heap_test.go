package hnsw

import (
	"cmp"
	"slices"
	"testing"
)

// cmpItem spells out (distance, ID) order independently of item.less, which it checks.
func cmpItem(a, b item) int {
	return cmp.Or(cmp.Compare(a.dist, b.dist), cmp.Compare(a.id, b.id))
}

// heapOps lets one oracle test drive both heaps.
type heapOps struct {
	name string
	push func(item)
	pop  func() item
	best func(oracle []item) int // index of the item the oracle expects next
}

func newHeapOps() []heapOps {
	var lo minHeap
	var hi maxHeap
	return []heapOps{
		{"min", lo.push, lo.pop, func(o []item) int { return 0 }},
		{"max", hi.push, hi.pop, func(o []item) int { return len(o) - 1 }},
	}
}

func TestHeapsMatchSortOracle(t *testing.T) {
	s := splitmix(3)
	draw := func(ties bool) item {
		if ties { // few distinct distances, so the ID tie-break decides
			return item{float32(s.next() % 4), uint32(s.next() % 64)}
		}
		return item{s.signed(), uint32(s.next())}
	}
	for _, h := range newHeapOps() {
		for _, ties := range []bool{false, true} {
			var oracle []item
			pop := func() {
				t.Helper()
				i := h.best(oracle)
				if got := h.pop(); got != oracle[i] {
					t.Fatalf("%s ties=%v: popped %v, want %v", h.name, ties, got, oracle[i])
				}
				oracle = slices.Delete(oracle, i, i+1)
			}
			for range 20000 {
				if len(oracle) == 0 || s.next()%3 != 0 {
					x := draw(ties)
					h.push(x)
					i, _ := slices.BinarySearchFunc(oracle, x, cmpItem)
					oracle = slices.Insert(oracle, i, x)
				} else {
					pop()
				}
			}
			for len(oracle) > 0 {
				pop()
			}
		}
	}
}

func TestHeapsAcrossSizes(t *testing.T) {
	s := splitmix(5)
	for n := range 70 {
		items := make([]item, n)
		for i := range items {
			items[i] = item{s.signed(), uint32(i)}
		}
		var lo minHeap
		var hi maxHeap
		for _, x := range items {
			lo.push(x)
			hi.push(x)
		}
		slices.SortFunc(items, cmpItem)
		for i := range items {
			if got := lo.pop(); got != items[i] {
				t.Fatalf("n=%d min[%d]: got %v, want %v", n, i, got, items[i])
			}
			if got := hi.pop(); got != items[n-1-i] {
				t.Fatalf("n=%d max[%d]: got %v, want %v", n, i, got, items[n-1-i])
			}
		}
		if len(lo) != 0 || len(hi) != 0 {
			t.Fatalf("n=%d: heaps not empty", n)
		}
	}
}

func TestHeapsDoNotAllocate(t *testing.T) {
	s := splitmix(9)
	items := make([]item, 1000)
	for i := range items {
		items[i] = item{s.signed(), uint32(i)}
	}
	lo, hi := make(minHeap, 0, len(items)), make(maxHeap, 0, len(items))
	if n := testing.AllocsPerRun(100, func() {
		for _, x := range items {
			lo.push(x)
			hi.push(x)
		}
		for range items {
			lo.pop()
			hi.pop()
		}
	}); n != 0 {
		t.Fatalf("%v allocations per run", n)
	}
}
