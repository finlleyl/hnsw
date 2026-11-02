//go:build !race

package hnsw

import (
	"math"
	"sync/atomic"
	"testing"
)

//go:noinline
func storeFloat32(p *float32, x float32) { *p = x }

// The race build excludes this because mutating src during normalize is the point. A
// single core can miss a regression here, but correct code cannot fail.
func TestNormalizeSurvivesConcurrentSrcMutation(t *testing.T) {
	src := make([]float32, 256)
	for i := range src {
		src[i] = 1
	}
	nan := float32(math.NaN())
	var stop atomic.Bool
	done := make(chan struct{})
	go func() {
		defer close(done)
		for !stop.Load() {
			storeFloat32(&src[100], nan)
			storeFloat32(&src[100], 1)
		}
	}()
	defer func() {
		stop.Store(true)
		<-done
	}()
	dst := make([]float32, len(src))
	for range 200000 {
		if normalize(dst, src) == nil && math.IsNaN(float64(dst[100])) {
			t.Fatal("NaN accepted")
		}
	}
}
