package hnsw

import (
	"context"
	"fmt"
)

// Compact rebuilds the graph from live nodes. Searches continue on the old graph, while
// Add, Delete and WriteTo block until the swap. Cancelling leaves the index unchanged.
func (ix *Index[K]) Compact(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("%w: nil context", ErrInvalidArgument)
	}
	ix.wmu.Lock()
	defer ix.wmu.Unlock()
	old := ix.g
	if old == nil {
		return fmt.Errorf("%w: zero Index", ErrInvalidArgument)
	}
	if old.deadN == 0 {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	next := &Index[K]{
		g:         newGraph[K](old.dim, old.m),
		dim:       ix.dim,
		efc:       ix.efc,
		ef:        ix.ef,
		ml:        ix.ml,
		seed:      ix.seed,
		stringKey: ix.stringKey,
	}
	inserted := 0
	add := func(key K, vector []float32) error {
		if inserted%1024 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if err := next.Add(key, vector); err != nil {
			return err
		}
		inserted++
		return nil
	}
	for id := uint32(0); id < old.n; id++ {
		if old.isDead(id) {
			continue
		}
		if err := add(old.keys[id], old.vec(id)); err != nil {
			return err
		}
		for _, key := range old.alias[id] {
			if err := add(key, old.vec(id)); err != nil {
				return err
			}
		}
	}
	ix.mu.Lock()
	ix.g = next.g
	ix.mu.Unlock()
	return nil
}
