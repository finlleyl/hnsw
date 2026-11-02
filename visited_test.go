package hnsw

import "testing"

func TestVisitedMatchesMapOracle(t *testing.T) {
	s := splitmix(11)
	var v visited
	// Sizes grow and shrink, so stale words and spare capacity both get exercised.
	for _, n := range []int{1, 63, 64, 65, 128, 1000, 10, 100000, 64, 1 << 20, 3, 1 << 20} {
		for range 3 {
			v.reset(n)
			seen := map[uint32]bool{}
			check := func(id uint32) {
				t.Helper()
				if got := v.visit(id); got == seen[id] {
					t.Fatalf("n=%d id=%d: visit returned %v", n, id, got)
				}
				seen[id] = true
			}
			check(0)
			check(uint32(n - 1))
			check(uint32(n - 1))
			for range min(2*n, 5000) {
				check(uint32(s.next() % uint64(n)))
			}
		}
	}
}

func TestVisitedResetClearsWithoutFinishing(t *testing.T) {
	var v visited
	v.reset(1000)
	v.visit(999)
	v.reset(1000) // the previous operation never cleaned up, as after a panic
	if !v.visit(999) {
		t.Fatal("stale bit survived reset")
	}
	v.reset(70) // stale word now sits in spare capacity
	v.reset(1000)
	if !v.visit(999) {
		t.Fatal("stale bit survived in spare capacity")
	}
}

func TestVisitedResetClearsOnlyTouchedWords(t *testing.T) {
	var v visited
	v.reset(1 << 16)
	v.visit(0)
	v.bits[1] = 1 // a word not in touched, which a full clear would wipe
	v.reset(1 << 16)
	if v.bits[0] != 0 || v.bits[1] != 1 {
		t.Fatal("reset must clear exactly the touched words")
	}
}

// AllocsPerRun's warm-up would hide a missing presize, so check capacity directly.
func TestVisitedTouchedStaysWithinCapacity(t *testing.T) {
	var v visited
	v.reset(1 << 16)
	c := cap(v.touched)
	for id := range uint32(1 << 16) {
		v.visit(id)
	}
	if len(v.touched) != len(v.bits) {
		t.Fatalf("touched %d words, have %d", len(v.touched), len(v.bits))
	}
	if cap(v.touched) != c {
		t.Fatalf("touched grew from %d to %d", c, cap(v.touched))
	}
}

func TestVisitedDoesNotAllocate(t *testing.T) {
	const n = 1 << 20
	s := splitmix(13)
	ids := make([]uint32, 5000)
	for i := range ids {
		ids[i] = uint32(s.next() % n)
	}
	var v visited
	v.reset(n)
	if allocs := testing.AllocsPerRun(100, func() {
		v.reset(n)
		for _, id := range ids {
			v.visit(id)
		}
	}); allocs != 0 {
		t.Fatalf("%v allocations per run", allocs)
	}
}
