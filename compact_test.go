package hnsw

import (
	"context"
	"errors"
	"iter"
	"slices"
	"testing"
)

func TestCompact(t *testing.T) {
	const n = 300
	vecs := clustered(n+1, 8, 44)
	ix := build(t, vecs[:n], Options{M: 4, EfConstruction: 40, Seed: 12})
	for key := 0; key < n; key += 10 {
		if err := ix.Add(n+key, vecs[key]); err != nil {
			t.Fatal(err)
		}
	}
	for key := 0; key < n; key += 3 {
		ix.Delete(key)
		if key%10 == 0 {
			ix.Delete(n + key)
		}
	}
	for key := 10; key < n; key += 20 {
		ix.Delete(key)
	}
	model := map[int][]float32{}
	for key := range ix.Keys() {
		model[key] = slices.Clone(ix.g.vec(ix.g.byKey[key]))
	}
	liveNodes := int(ix.g.live)
	old := ix.g
	if err := ix.Compact(context.Background()); err != nil {
		t.Fatal(err)
	}
	if ix.g == old {
		t.Fatal("Compact did not replace the graph")
	}
	if s := ix.Stats(); s.Nodes != liveNodes || s.Deleted != 0 || s.Keys != len(model) {
		t.Fatalf("stats after Compact: %+v, want %d nodes and %d keys", s, liveNodes, len(model))
	}
	if err := ix.g.verify(); err != nil {
		t.Fatal(err)
	}
	if got := ix.g.unreached(); got != 0 {
		t.Fatalf("%d live nodes unreached", got)
	}
	for key, want := range model {
		if !ix.Contains(key) {
			t.Fatalf("key %d disappeared", key)
		}
		if got := ix.g.vec(ix.g.byKey[key]); !sameBits(got, want) {
			t.Fatalf("key %d is served by a different vector", key)
		}
	}
	res, err := ix.Search(vecs[1], 20, SearchOptions{Ef: n})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range res {
		if _, ok := model[r.Key]; !ok {
			t.Fatalf("deleted key %d returned", r.Key)
		}
	}
	checkRoundTrip(t, ix)
	if err := ix.Add(n*2, vecs[n]); err != nil {
		t.Fatal(err)
	}
	if !ix.Delete(1) {
		t.Fatal("Delete after Compact failed")
	}
	if err := ix.Compact(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := ix.g.verify(); err != nil {
		t.Fatal(err)
	}
}

func TestCompactNoOpAndCancel(t *testing.T) {
	vecs := clustered(20, 4, 51)
	ix := build(t, vecs, Options{M: 4})
	old := ix.g
	before := persistenceBytes(t, ix)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := ix.Compact(cancelled); err != nil {
		t.Fatalf("no-op Compact: %v", err)
	}
	if ix.g != old {
		t.Fatal("no-op Compact replaced the graph")
	}
	if err := ix.Compact(nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil context: %v", err)
	}
	ix.Delete(0)
	before = persistenceBytes(t, ix)
	if err := ix.Compact(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled Compact: %v", err)
	}
	if ix.g != old {
		t.Fatal("cancelled Compact replaced the graph")
	}
	if after := persistenceBytes(t, ix); !slices.Equal(after, before) {
		t.Fatal("cancelled Compact changed the saved bytes")
	}
	var zero Index[int]
	if err := zero.Compact(context.Background()); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("zero Index: %v", err)
	}
}

func TestCompactAllDeleted(t *testing.T) {
	vecs := clustered(10, 4, 52)
	ix := build(t, vecs, Options{M: 2})
	for key := range vecs {
		ix.Delete(key)
	}
	if err := ix.Compact(context.Background()); err != nil {
		t.Fatal(err)
	}
	if s := ix.Stats(); s.Nodes != 0 || s.Keys != 0 || s.Deleted != 0 || s.MaxLevel != 0 {
		t.Fatalf("empty stats after Compact: %+v", s)
	}
	if ix.g.entry != noEntry || ix.Len() != 0 {
		t.Fatal("all-deleted Compact did not empty the graph")
	}
	checkRoundTrip(t, ix)
	if err := ix.Add(99, vecs[0]); err != nil {
		t.Fatal(err)
	}
	if !ix.Contains(99) {
		t.Fatal("Add after emptying Compact failed")
	}
}

func TestKeysPinsGraphAcrossCompact(t *testing.T) {
	vecs := clustered(4, 4, 53)
	ix := build(t, vecs, Options{M: 2})
	ix.Delete(3)
	old := ix.g
	next, stop := iter.Pull(ix.Keys())
	defer stop()
	key, ok := next()
	if !ok || key != 0 {
		t.Fatalf("first key %d, %v", key, ok)
	}
	if err := ix.Compact(context.Background()); err != nil {
		t.Fatal(err)
	}
	old.keys[1] = 99
	key, ok = next()
	if !ok || key != 99 {
		t.Fatalf("iterator switched graphs: key %d, %v", key, ok)
	}
}
