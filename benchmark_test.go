package hnsw

import (
	"fmt"
	"testing"
)

func BenchmarkSearch(b *testing.B) {
	const n, nq = 50000, 1000
	for _, d := range []int{128, 768, 1536} {
		b.Run(fmt.Sprintf("d=%d", d), func(b *testing.B) {
			all := clustered(n+nq, d, 7)
			ix, err := New[int](d, Options{M: 16, Seed: 7})
			if err != nil {
				b.Fatal(err)
			}
			for key := range n {
				if err := ix.Add(key, all[key]); err != nil {
					b.Fatal(err)
				}
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := ix.Search(all[n+i%nq], 10, SearchOptions{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkAdd(b *testing.B) {
	const n = 10000
	for _, d := range []int{128, 768, 1536} {
		b.Run(fmt.Sprintf("d=%d", d), func(b *testing.B) {
			all := clustered(n+b.N, d, 7)
			ix, err := New[int](d, Options{M: 16, Seed: 7})
			if err != nil {
				b.Fatal(err)
			}
			for key := range n {
				if err := ix.Add(key, all[key]); err != nil {
					b.Fatal(err)
				}
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := ix.Add(n+i, all[n+i]); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
