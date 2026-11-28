package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"slices"
	"sync"
	"time"

	"github.com/finlleyl/hnsw"
)

type compactResult struct {
	Method      string  `json:"method"`
	Dim         int     `json:"dim"`
	M           int     `json:"m"`
	Keys        int     `json:"keys"`
	NodesBefore int     `json:"nodes_before"`
	NodesAfter  int     `json:"nodes_after"`
	Seconds     float64 `json:"seconds"`
	Ef          int     `json:"ef"`
	RecallAt10  float64 `json:"recall_at_10"`
	MSPerQuery  float64 `json:"ms_per_query"`
}

func deletedIndexPath() string {
	return indexPath(768, 16) + ".deleted30"
}

func liveTruthPath() string {
	return fmt.Sprintf("%s.live30", truthPath(768))
}

func compactRebuild() error {
	if *dim != 768 || *m != 16 {
		return fmt.Errorf("compact gate requires d=768 and M=16")
	}
	if err := checkRaw(*dim); err != nil {
		return err
	}
	vectors, err := readFloats(rawPath(768), *rows*768)
	if err != nil {
		return err
	}
	queryVectors := slices.Clone(vectors[:*queries*768])
	base := vectors[*queries*768:]
	deleted := make([]bool, *rows-*queries)
	for key := range deleted {
		deleted[key] = key%10 < 3
	}
	started := time.Now()
	limits := liveTruth(base, queryVectors, deleted)
	if err := writeFloatsAtomic(liveTruthPath(), limits); err != nil {
		return err
	}
	log.Printf("live exact truth in %.1fs", time.Since(started).Seconds())
	vectors = nil
	base = nil
	runtime.GC()
	ix, err := loadDeletedIndex(deleted)
	if err != nil {
		return err
	}
	nodesBefore := ix.Stats().Nodes
	started = time.Now()
	if err := ix.Compact(context.Background()); err != nil {
		return err
	}
	seconds := time.Since(started).Seconds()
	recall, ms, err := liveRecall(ix, queryVectors, limits, 100)
	if err != nil {
		return err
	}
	stats := ix.Stats()
	out := compactResult{"rebuild", 768, 16, stats.Keys, nodesBefore, stats.Nodes, seconds, 100, recall, ms}
	return json.NewEncoder(os.Stdout).Encode(out)
}

func loadDeletedIndex(deleted []bool) (*hnsw.Index[int], error) {
	ix, err := hnsw.LoadFile[int](indexPath(768, 16))
	if err != nil {
		return nil, err
	}
	for key, remove := range deleted {
		if remove {
			ix.Delete(key)
		}
	}
	if err := ix.SaveFile(deletedIndexPath()); err != nil {
		return nil, err
	}
	return ix, nil
}

func liveTruth(base, queryVectors []float32, deleted []bool) []float32 {
	limits := make([]float32, *queries)
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for q := w; q < *queries; q += workers {
				query := queryVectors[q*768 : (q+1)*768]
				limits[q] = liveTruthLimit(base, query, deleted)
			}
		}()
	}
	wg.Wait()
	return limits
}

func liveTruthLimit(base, query []float32, deleted []bool) float32 {
	const k = 10
	best := make([]float32, k)
	for i := range best {
		best[i] = float32(math.Inf(1))
	}
	for key := range deleted {
		if deleted[key] {
			continue
		}
		v := base[key*768 : (key+1)*768]
		distance := 1 - dot(query, v)
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

func liveRecall(ix *hnsw.Index[int], queryVectors []float32, limits []float32, ef int) (float64, float64, error) {
	hits := 0
	started := time.Now()
	for q := range *queries {
		query := queryVectors[q*768 : (q+1)*768]
		res, err := ix.Search(query, 10, hnsw.SearchOptions{Ef: ef})
		if err != nil {
			return 0, 0, err
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
	return recall, float64(elapsed) / float64(time.Millisecond) / float64(*queries), nil
}
