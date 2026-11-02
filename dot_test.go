package hnsw

import (
	"fmt"
	"math"
	"testing"
)

func TestDotCountsEveryElement(t *testing.T) {
	for d := range 34 {
		for i := range d {
			a, b := make([]float32, d), make([]float32, d)
			a[i], b[i] = 2, 3
			if got := dot(a, b); got != 6 {
				t.Fatalf("d=%d i=%d: got %v, want 6", d, i, got)
			}
		}
	}
	if got := dot(nil, nil); got != 0 {
		t.Fatalf("empty: got %v", got)
	}
}

func TestDotMatchesFloat64Reference(t *testing.T) {
	s := splitmix(7)
	for _, d := range testDims {
		for range 50 {
			a, b := make([]float32, d), make([]float32, d)
			for i := range a {
				a[i], b[i] = s.signed(), s.signed()
			}
			if normalize(a, a) != nil || normalize(b, b) != nil {
				continue
			}
			var want float64
			for i := range a {
				want += float64(a[i]) * float64(b[i])
			}
			if got := float64(dot(a, b)); math.Abs(got-want) > 1e-5 {
				t.Fatalf("d=%d: got %v, want %v", d, got, want)
			}
			if self := float64(dot(a, a)); math.Abs(self-1) > 1e-5 {
				t.Fatalf("d=%d: self dot %v", d, self)
			}
		}
	}
}

var dotSink float32

func BenchmarkDot(b *testing.B) {
	s := splitmix(1)
	for _, d := range []int{128, 768, 1536, 3072} {
		x, y := make([]float32, d), make([]float32, d)
		for i := range x {
			x[i], y[i] = s.signed(), s.signed()
		}
		b.Run(fmt.Sprintf("d=%d", d), func(b *testing.B) {
			for b.Loop() {
				dotSink = dot(x, y)
			}
		})
	}
}
