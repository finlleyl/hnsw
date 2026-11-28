package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
)

var (
	dataDir = flag.String("data", "data", "measurement data directory")
	mode    = flag.String("mode", "run", "download, prepare, truth, build, run, search, or compact")
	dim     = flag.Int("d", 768, "768, 1536, or 3072")
	m       = flag.Int("m", 16, "HNSW M")
	rows    = flag.Int("rows", 1000000, "rows including held-out queries")
	queries = flag.Int("queries", 1000, "held-out queries at the start of the dataset")
	shards  = flag.Int("shards", 63, "Parquet shards to download")
)

func main() {
	flag.Parse()
	log.SetOutput(os.Stderr)
	if *rows < 2 || *queries < 1 || *queries >= *rows {
		log.Fatal("invalid rows or queries")
	}
	var err error
	switch *mode {
	case "download":
		err = download()
	case "prepare":
		err = prepare()
	case "truth":
		err = makeTruth()
	case "build":
		_, err = buildIndex()
	case "run":
		var ix *index
		ix, err = buildIndex()
		if err == nil {
			err = searchIndex(ix.ix)
		}
	case "search":
		err = searchSaved()
	case "compact":
		err = compactRebuild()
	default:
		log.Fatalf("unknown mode %q", *mode)
	}
	if err != nil {
		log.Fatal(err)
	}
}

func validDim(d int) bool { return d == 768 || d == 1536 || d == 3072 }

func rawPath(d int) string {
	return filepath.Join(*dataDir, fmt.Sprintf("dbpedia-d%d.f32", d))
}

func truthPath(d int) string {
	return filepath.Join(*dataDir, fmt.Sprintf("truth-d%d-q%d.f32", d, *queries))
}

func indexPath(d, m int) string {
	return filepath.Join(*dataDir, fmt.Sprintf("index-d%d-m%d.hnsw", d, m))
}

func checkRaw(d int) error {
	if !validDim(d) {
		return fmt.Errorf("invalid dimension %d", d)
	}
	st, err := os.Stat(rawPath(d))
	if err != nil {
		return err
	}
	want := int64(*rows) * int64(d) * 4
	if st.Size() != want {
		return fmt.Errorf("%s is %d bytes, want %d", rawPath(d), st.Size(), want)
	}
	return nil
}

func readFloats(path string, count int) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := make([]float32, count)
	if err := binary.Read(f, binary.LittleEndian, out); err != nil {
		return nil, err
	}
	return out, nil
}

func readVector(r *bufio.Reader, buf []byte, v []float32) error {
	if _, err := io.ReadFull(r, buf); err != nil {
		return err
	}
	n, err := binary.Decode(buf, binary.LittleEndian, &v)
	if err != nil {
		return err
	}
	if n != len(buf) {
		return fmt.Errorf("decoded %d of %d bytes", n, len(buf))
	}
	return nil
}

func writeFloatsAtomic(path string, values []float32) error {
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	if err := binary.Write(w, binary.LittleEndian, values); err == nil {
		err = w.Flush()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

func normalize(v []float32) error {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if math.IsNaN(sum) || math.IsInf(sum, 0) || sum == 0 {
		return fmt.Errorf("invalid vector")
	}
	if math.Abs(sum-1) > 1e-6 {
		scale := math.Sqrt(sum)
		for i, x := range v {
			v[i] = float32(float64(x) / scale)
		}
	}
	for i, x := range v {
		if x == 0 {
			v[i] = 0
		}
	}
	return nil
}

func dot(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	for len(a) >= 8 {
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
