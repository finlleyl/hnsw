package main

import (
	"encoding/json"
	"flag"
	"log"
	"math"
	"math/rand"
	rand2 "math/rand/v2"
	"os"
	"runtime"
	"slices"
	"sync"
	"testing"
	"time"

	coder "github.com/coder/hnsw"
	ours "github.com/finlleyl/hnsw"
)

var (
	mode    = flag.String("mode", "recall", "recall or kernel")
	impl    = flag.String("impl", "ours", "ours or coder")
	dim     = flag.Int("d", 0, "dimension; 0 measures 768 and 1536")
	n       = flag.Int("n", 50000, "base vectors")
	nq      = flag.Int("queries", 1000, "held-out queries")
	seconds = flag.Float64("seconds", 5, "measurement seconds per ef")
	sink    int
)

type index interface {
	Search([]float32, int, int) ([]int, error)
	Nodes() int
}

type ourIndex struct{ ix *ours.Index[int] }

func (i ourIndex) Search(q []float32, k, ef int) ([]int, error) {
	res, err := i.ix.Search(q, k, ours.SearchOptions{Ef: ef})
	if err != nil {
		return nil, err
	}
	out := make([]int, len(res))
	for j, r := range res {
		out[j] = r.Key
	}
	return out, nil
}
func (i ourIndex) Nodes() int { return i.ix.Stats().Nodes }

type coderIndex struct{ g *coder.Graph[int] }

func (i coderIndex) Search(q []float32, k, ef int) ([]int, error) {
	i.g.EfSearch = ef
	res := i.g.Search(q, k)
	out := make([]int, len(res))
	for j, r := range res {
		out[j] = r.Key
	}
	return out, nil
}
func (i coderIndex) Nodes() int { return i.g.Len() }

type recallResult struct {
	Implementation string  `json:"implementation"`
	Dim            int     `json:"dim"`
	N              int     `json:"n"`
	Nodes          int     `json:"nodes"`
	BuildSeconds   float64 `json:"build_seconds"`
	Ef             int     `json:"ef"`
	RecallAt10     float64 `json:"recall_at_10"`
	QPS            float64 `json:"qps"`
}

func main() {
	flag.Parse()
	log.SetOutput(os.Stderr)
	if *mode == "kernel" {
		runKernels()
		return
	}
	if *mode != "recall" || (*impl != "ours" && *impl != "coder") || *n < 1 || *nq < 1 || *seconds <= 0 {
		log.Fatal("invalid flags")
	}
	dims := []int{*dim}
	if *dim == 0 {
		dims = []int{768, 1536}
	}
	for _, d := range dims {
		if err := runRecall(d); err != nil {
			log.Fatal(err)
		}
		runtime.GC()
	}
}

func runRecall(d int) error {
	all := clustered(*n+*nq, d, 7)
	base, queries := all[:*n], all[*n:]
	started := time.Now()
	ix, err := build(*impl, base)
	if err != nil {
		return err
	}
	buildSeconds := time.Since(started).Seconds()
	log.Printf("%s d=%d: built %d nodes in %.1fs", *impl, d, ix.Nodes(), buildSeconds)
	truthStarted := time.Now()
	limits := truthLimits(base, queries, 10)
	log.Printf("%s d=%d: exact truth in %.1fs", *impl, d, time.Since(truthStarted).Seconds())
	runtime.GOMAXPROCS(1)
	for _, ef := range []int{20, 40, 80, 160, 320} {
		started = time.Now()
		calls := 0
		for time.Since(started) < time.Duration(*seconds*float64(time.Second)) {
			res, err := ix.Search(queries[calls%len(queries)], 10, ef)
			if err != nil {
				return err
			}
			sink += len(res)
			calls++
		}
		elapsed := time.Since(started)
		hits := 0
		for q, query := range queries {
			res, err := ix.Search(query, 10, ef)
			if err != nil {
				return err
			}
			limit := max(0, limits[q]) + 2e-6
			for _, key := range res {
				if distance(query, base[key]) <= limit {
					hits++
				}
			}
		}
		out := recallResult{
			Implementation: *impl, Dim: d, N: *n, Nodes: ix.Nodes(),
			BuildSeconds: buildSeconds, Ef: ef,
			RecallAt10: float64(hits) / float64(10*len(queries)),
			QPS:        float64(calls) / elapsed.Seconds(),
		}
		if err := json.NewEncoder(os.Stdout).Encode(out); err != nil {
			return err
		}
	}
	return nil
}

func build(name string, vecs [][]float32) (index, error) {
	if name == "ours" {
		ix, err := ours.New[int](len(vecs[0]), ours.Options{M: 16, Seed: 7})
		if err != nil {
			return nil, err
		}
		for key, v := range vecs {
			if err := ix.Add(key, v); err != nil {
				return nil, err
			}
			if key%10000 == 9999 {
				log.Printf("added %d/%d", key+1, len(vecs))
			}
		}
		return ourIndex{ix}, nil
	}
	g := coder.NewGraph[int]()
	g.M = 16
	g.Ml = 1 / math.Log(16)
	g.EfSearch = 200
	g.Rng = rand.New(rand.NewSource(7))
	for key, v := range vecs {
		g.Add(coder.MakeNode(key, v))
		if key%10000 == 9999 {
			log.Printf("added %d/%d", key+1, len(vecs))
		}
	}
	return coderIndex{g}, nil
}

func truthLimits(base, queries [][]float32, k int) []float32 {
	out := make([]float32, len(queries))
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for q := w; q < len(queries); q += workers {
				best := make([]float32, k)
				for i := range best {
					best[i] = float32(math.Inf(1))
				}
				for _, v := range base {
					d := distance(queries[q], v)
					if d >= best[k-1] {
						continue
					}
					pos := k - 1
					for pos > 0 && best[pos-1] > d {
						best[pos] = best[pos-1]
						pos--
					}
					best[pos] = d
				}
				out[q] = best[k-1]
			}
		}()
	}
	wg.Wait()
	return out
}

func distance(a, b []float32) float32 { return 1 - dot8(a, b) }

func normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	scale := math.Sqrt(sum)
	for i, x := range v {
		v[i] = float32(float64(x) / scale)
		if v[i] == 0 {
			v[i] = 0
		}
	}
}

func clustered(n, d int, seed uint64) [][]float32 {
	const nc, rank = 256, 16
	r := rand2.New(rand2.NewPCG(seed, 2))
	centres := make([][]float32, nc)
	bases := make([][]float32, nc)
	cdf := make([]float64, nc)
	total := 0.0
	for c := range nc {
		centres[c] = make([]float32, d)
		for i := range centres[c] {
			centres[c][i] = float32(r.NormFloat64())
		}
		normalize(centres[c])
		spread := 0.1 + 0.5*r.Float64()
		bases[c] = make([]float32, rank*d)
		for i := range bases[c] {
			bases[c][i] = float32(r.NormFloat64() * spread / math.Sqrt(float64(d)*rank))
		}
		total += 1 / float64(c+1)
		cdf[c] = total
	}
	noise := 0.02 / math.Sqrt(float64(d))
	out := make([][]float32, n)
	for i := range out {
		u := r.Float64() * total
		c := 0
		for cdf[c] < u {
			c++
		}
		v := slices.Clone(centres[c])
		for j := range rank {
			z := float32(r.NormFloat64())
			for k, b := range bases[c][j*d : (j+1)*d] {
				v[k] += z * b
			}
		}
		for k := range v {
			v[k] += float32(r.NormFloat64() * noise)
		}
		normalize(v)
		out[i] = v
	}
	return out
}

type kernelResult struct {
	Kernel string  `json:"kernel"`
	Dim    int     `json:"dim"`
	Run    int     `json:"run"`
	NS     float64 `json:"ns_per_op"`
}

func runKernels() {
	kernels := map[string]func([]float32, []float32) float32{
		"scalar": dotScalar, "unrolled4": dot4, "unrolled8": dot8, "unrolled16": dot16,
	}
	r := rand2.New(rand2.NewPCG(1, 2))
	for _, d := range []int{128, 768, 1536, 3072} {
		a, b := make([]float32, d), make([]float32, d)
		for i := range a {
			a[i], b[i] = float32(r.NormFloat64()), float32(r.NormFloat64())
		}
		normalize(a)
		normalize(b)
		var want float64
		for i := range a {
			want += float64(a[i]) * float64(b[i])
		}
		names := []string{"scalar", "unrolled4", "unrolled8", "unrolled16"}
		for _, name := range names {
			kernel := kernels[name]
			if got := float64(kernel(a, b)); math.Abs(got-want) > 1e-5 {
				log.Fatalf("%s d=%d: got %g, want %g", name, d, got, want)
			}
			for run := range 3 {
				res := testing.Benchmark(func(bt *testing.B) {
					for bt.Loop() {
						sink = int(kernel(a, b))
					}
				})
				json.NewEncoder(os.Stdout).Encode(kernelResult{name, d, run, float64(res.T.Nanoseconds()) / float64(res.N)})
			}
		}
	}
}

func dotScalar(a, b []float32) float32 {
	var s float32
	for i, x := range a {
		s += x * b[i]
	}
	return s
}

func dot4(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	for len(a) >= 4 {
		s0 += a[0] * b[0]
		s1 += a[1] * b[1]
		s2 += a[2] * b[2]
		s3 += a[3] * b[3]
		a, b = a[4:], b[4:]
	}
	for i, x := range a {
		s0 += x * b[i]
	}
	return s0 + s1 + s2 + s3
}

func dot8(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	for len(a) >= 8 && len(b) >= 8 {
		s0 += a[0] * b[0]
		s1 += a[1] * b[1]
		s2 += a[2] * b[2]
		s3 += a[3] * b[3]
		s4 += a[4] * b[4]
		s5 += a[5] * b[5]
		s6 += a[6] * b[6]
		s7 += a[7] * b[7]
		a, b = a[8:], b[8:]
	}
	for i, x := range a {
		s0 += x * b[i]
	}
	return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
}

func dot16(a, b []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	var s8, s9, s10, s11, s12, s13, s14, s15 float32
	for len(a) >= 16 {
		s0 += a[0] * b[0]
		s1 += a[1] * b[1]
		s2 += a[2] * b[2]
		s3 += a[3] * b[3]
		s4 += a[4] * b[4]
		s5 += a[5] * b[5]
		s6 += a[6] * b[6]
		s7 += a[7] * b[7]
		s8 += a[8] * b[8]
		s9 += a[9] * b[9]
		s10 += a[10] * b[10]
		s11 += a[11] * b[11]
		s12 += a[12] * b[12]
		s13 += a[13] * b[13]
		s14 += a[14] * b[14]
		s15 += a[15] * b[15]
		a, b = a[16:], b[16:]
	}
	for i, x := range a {
		s0 += x * b[i]
	}
	return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12 + s13 + s14 + s15
}
