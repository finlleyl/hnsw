package hnsw

import (
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"
)

func TestConcurrentContract(t *testing.T) {
	vecs := clustered(512, 8, 19)
	ix := build(t, vecs[:256], Options{M: 4, EfConstruction: 40})
	type deletion struct {
		key int
		v   []float32
	}
	deletes := make(chan deletion, 64)
	done := make(chan struct{})
	timer := time.AfterFunc(3*time.Second, func() { close(done) })
	defer timer.Stop()
	var wg sync.WaitGroup

	const adders = 4
	for worker := range adders {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; ; i++ {
				key := 1000 + worker + adders*i
				v := vecs[key%len(vecs)]
				if err := ix.Add(key, v); err != nil {
					t.Errorf("Add(%d): %v", key, err)
					return
				}
				if !ix.Contains(key) {
					t.Errorf("Contains(%d) is false after Add returned", key)
					return
				}
				select {
				case deletes <- deletion{key, v}:
				case <-done:
					return
				}
			}
		}()
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case d := <-deletes:
				if !ix.Delete(d.key) {
					t.Errorf("Delete(%d) is false after Add returned", d.key)
					return
				}
				res, err := ix.Search(d.v, 20, SearchOptions{})
				if err != nil {
					t.Errorf("Search after Delete(%d): %v", d.key, err)
					return
				}
				for _, r := range res {
					if r.Key == d.key {
						t.Errorf("Search after Delete returned key %d", d.key)
						return
					}
				}
			case <-done:
				return
			}
		}
	}()

	for worker := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; ; i++ {
				res, err := ix.Search(vecs[(worker+i)%len(vecs)], 10, SearchOptions{})
				if err != nil {
					t.Errorf("Search: %v", err)
					return
				}
				for j, r := range res {
					if j > 0 && r.Distance < res[j-1].Distance {
						t.Errorf("results are not sorted: %v", res)
						return
					}
					for _, previous := range res[:j] {
						if previous.Key == r.Key {
							t.Errorf("Search returned key %d twice", r.Key)
							return
						}
					}
				}
				select {
				case <-done:
					return
				default:
				}
			}
		}()
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			for key := range ix.Keys() {
				_ = ix.Contains(key)
				_ = ix.Len()
				_ = ix.Stats()
				select {
				case <-done:
					return
				default:
				}
			}
		}
	}()
	wg.Wait()
	if err := ix.g.verify(); err != nil {
		t.Fatal(err)
	}
}

func TestSearchAllocatesOnlyResults(t *testing.T) {
	if raceEnabled {
		t.Skip("the race detector adds allocations inside sync.RWMutex")
	}
	vecs := clustered(1000, 32, 23)
	ix := build(t, vecs, Options{M: 4, EfConstruction: 40})
	var out []Result[int]
	allocs := testing.AllocsPerRun(100, func() {
		var err error
		out, err = ix.Search(vecs[7], 10, SearchOptions{})
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 1 {
		t.Fatalf("%v allocations per Search, want only the result slice", allocs)
	}
	if len(out) != 10 {
		t.Fatalf("%d results, want 10", len(out))
	}
}

func BenchmarkReaders100Adds(b *testing.B) {
	configs := []struct {
		n, d int
	}{{100000, 128}, {50000, 768}}
	for _, tc := range configs {
		for _, whole := range []bool{false, true} {
			mode := "split"
			if whole {
				mode = "whole"
			}
			b.Run(fmt.Sprintf("%s/d=%d", mode, tc.d), func(b *testing.B) {
				wholeInsert = whole
				defer func() {
					wholeInsert = false
					publishObserver = nil
				}()
				const queryCount, adds = 1000, 100
				all := clustered(tc.n+queryCount+adds*b.N, tc.d, 7)
				ix, err := New[int](tc.d, Options{M: 16, Seed: 7})
				if err != nil {
					b.Fatal(err)
				}
				for key := range tc.n {
					if err := ix.Add(key, all[key]); err != nil {
						b.Fatal(err)
					}
				}
				var samples []time.Duration
				var maxPublish time.Duration
				publishObserver = func(d time.Duration) {
					if d > maxPublish {
						maxPublish = d
					}
				}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					stop := make(chan struct{})
					durations := make(chan time.Duration, 1<<20)
					var wg sync.WaitGroup
					for worker := range 8 {
						wg.Add(1)
						go func() {
							defer wg.Done()
							for j := 0; ; j++ {
								started := time.Now()
								_, err := ix.Search(all[tc.n+(worker+j)%queryCount], 10, SearchOptions{})
								elapsed := time.Since(started)
								if err != nil {
									b.Errorf("Search: %v", err)
									return
								}
								select {
								case durations <- elapsed:
								case <-stop:
									return
								}
								select {
								case <-stop:
									return
								default:
								}
							}
						}()
					}
					started := time.Now()
					for j := 0; j < adds; j++ {
						delay := time.Until(started.Add(time.Duration(j) * time.Second / adds))
						if delay > 0 {
							time.Sleep(delay)
						}
						key := tc.n + i*adds + j
						if err := ix.Add(key, all[tc.n+queryCount+i*adds+j]); err != nil {
							b.Errorf("Add: %v", err)
							break
						}
					}
					close(stop)
					wg.Wait()
					close(durations)
					for d := range durations {
						samples = append(samples, d)
					}
				}
				b.StopTimer()
				publishObserver = nil
				if len(samples) == 0 {
					b.Fatal("no Search samples")
				}
				slices.Sort(samples)
				b.ReportMetric(float64(samples[len(samples)*99/100]), "search_p99_ns")
				b.ReportMetric(float64(maxPublish), "publish_max_ns")
			})
		}
	}
}
