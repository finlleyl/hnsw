package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/finlleyl/hnsw"
)

type index struct {
	ix *hnsw.Index[int]
}

type buildResult struct {
	Dim          int     `json:"dim"`
	M            int     `json:"m"`
	Keys         int     `json:"keys"`
	Nodes        int     `json:"nodes"`
	BuildSeconds float64 `json:"build_seconds"`
	SaveSeconds  float64 `json:"save_seconds"`
}

type searchResult struct {
	Dim        int     `json:"dim"`
	M          int     `json:"m"`
	Keys       int     `json:"keys"`
	Nodes      int     `json:"nodes"`
	Ef         int     `json:"ef"`
	RecallAt10 float64 `json:"recall_at_10"`
	MSPerQuery float64 `json:"ms_per_query"`
}

func makeTruth() error {
	if err := checkRaw(*dim); err != nil {
		return err
	}
	started := time.Now()
	vectors, err := readFloats(rawPath(*dim), *rows**dim)
	if err != nil {
		return err
	}
	log.Printf("loaded %d normalized vectors in %.1fs", *rows, time.Since(started).Seconds())
	base := vectors[*queries**dim:]
	limits := make([]float32, *queries)
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for q := w; q < *queries; q += workers {
				query := vectors[q**dim : (q+1)**dim]
				limits[q] = truthLimit(base, query, *dim, 10)
				if q%100 == 99 {
					log.Printf("exact truth: query %d/%d", q+1, *queries)
				}
			}
		}()
	}
	wg.Wait()
	if err := writeFloatsAtomic(truthPath(*dim), limits); err != nil {
		return err
	}
	log.Printf("exact truth finished in %.1fs", time.Since(started).Seconds())
	return nil
}

func truthLimit(base, query []float32, d, k int) float32 {
	best := make([]float32, k)
	for i := range best {
		best[i] = float32(math.Inf(1))
	}
	for offset := 0; offset < len(base); offset += d {
		distance := 1 - dot(query, base[offset:offset+d])
		if distance >= best[k-1] {
			continue
		}
		pos := k - 1
		for pos > 0 && best[pos-1] > distance {
			best[pos] = best[pos-1]
			pos--
		}
		best[pos] = distance
	}
	return best[k-1]
}

func buildIndex() (*index, error) {
	if err := checkRaw(*dim); err != nil {
		return nil, err
	}
	ix, err := hnsw.New[int](*dim, hnsw.Options{M: *m, Seed: 7})
	if err != nil {
		return nil, err
	}
	f, err := os.Open(rawPath(*dim))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)
	buf := make([]byte, *dim*4)
	v := make([]float32, *dim)
	for range *queries {
		if err := readVector(r, buf, v); err != nil {
			return nil, err
		}
	}
	base := *rows - *queries
	started := time.Now()
	for key := range base {
		if err := readVector(r, buf, v); err != nil {
			return nil, fmt.Errorf("key %d: %w", key, err)
		}
		if err := ix.Add(key, v); err != nil {
			return nil, fmt.Errorf("key %d: %w", key, err)
		}
		if key%10000 == 9999 {
			log.Printf("d=%d M=%d: added %d/%d in %.1fs", *dim, *m, key+1, base, time.Since(started).Seconds())
		}
	}
	buildSeconds := time.Since(started).Seconds()
	saveStarted := time.Now()
	if err := ix.SaveFile(indexPath(*dim, *m)); err != nil {
		return nil, err
	}
	stats := ix.Stats()
	out := buildResult{*dim, *m, stats.Keys, stats.Nodes, buildSeconds, time.Since(saveStarted).Seconds()}
	return &index{ix}, json.NewEncoder(os.Stdout).Encode(out)
}

func searchSaved() error {
	started := time.Now()
	ix, err := hnsw.LoadFile[int](indexPath(*dim, *m))
	if err != nil {
		return err
	}
	log.Printf("loaded index in %.1fs", time.Since(started).Seconds())
	return searchIndex(ix)
}

func searchIndex(ix *hnsw.Index[int]) error {
	if err := checkRaw(*dim); err != nil {
		return err
	}
	queryVectors, err := readFloats(rawPath(*dim), *queries**dim)
	if err != nil {
		return err
	}
	limits, err := readFloats(truthPath(*dim), *queries)
	if err != nil {
		return err
	}
	stats := ix.Stats()
	if stats.Keys != *rows-*queries {
		return fmt.Errorf("index has %d keys, want %d", stats.Keys, *rows-*queries)
	}
	for _, ef := range []int{100, 160, 240, 320, 480, 640, 960, 1280, 1920, 3072, 4096, 8192, 16384, 65536} {
		hits := 0
		started := time.Now()
		for q := range *queries {
			query := queryVectors[q**dim : (q+1)**dim]
			res, err := ix.Search(query, 10, hnsw.SearchOptions{Ef: ef})
			if err != nil {
				return err
			}
			limit := max(0, limits[q]) + 2e-6
			for _, r := range res {
				if r.Distance <= limit {
					hits++
				}
			}
		}
		elapsed := time.Since(started)
		recall := float64(hits) / float64(10**queries)
		msPerQuery := float64(elapsed) / float64(time.Millisecond) / float64(*queries)
		out := searchResult{*dim, *m, stats.Keys, stats.Nodes, ef, recall, msPerQuery}
		if err := json.NewEncoder(os.Stdout).Encode(out); err != nil {
			return err
		}
		log.Printf("d=%d M=%d ef=%d: recall@10 %.4f", *dim, *m, ef, recall)
		if recall >= .99 {
			return nil
		}
	}
	return fmt.Errorf("recall@10 below .99 at ef=65536")
}
