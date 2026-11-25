package hnsw

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"testing"
)

func persistenceBytes[K Key](t testing.TB, ix *Index[K]) []byte {
	t.Helper()
	var buf bytes.Buffer
	n, err := ix.WriteTo(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if n != int64(buf.Len()) {
		t.Fatalf("WriteTo returned %d bytes, wrote %d", n, buf.Len())
	}
	return slices.Clone(buf.Bytes())
}

func checkRoundTrip[K Key](t *testing.T, ix *Index[K]) *Index[K] {
	t.Helper()
	data := persistenceBytes(t, ix)
	got, err := Load[K](bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if got.Stats() != ix.Stats() {
		t.Fatalf("stats after Load: %+v, want %+v", got.Stats(), ix.Stats())
	}
	if got.dim != ix.dim || got.efc != ix.efc || got.ef != ix.ef || got.seed != ix.seed {
		t.Fatalf("options were not preserved")
	}
	if err := got.g.verify(); err != nil {
		t.Fatal(err)
	}
	again := persistenceBytes(t, got)
	if !bytes.Equal(again, data) {
		t.Fatal("Save, Load, Save is not byte-identical")
	}
	return got
}

func persistenceIndex[K Key](t testing.TB, keys [3]K) *Index[K] {
	t.Helper()
	ix, err := New[K](2, Options{M: 4, EfConstruction: 40, Ef: 17, Seed: 99})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range [][]float32{{1, 0}, {1, 0}, {0, 1}, {-1, 0}} {
		key := keys[i%3]
		if err := ix.Add(key, v); err != nil {
			t.Fatal(err)
		}
	}
	if !ix.Delete(keys[2]) {
		t.Fatal("Delete failed")
	}
	return ix
}

func checkPersistenceKeys[K Key](t *testing.T, keys [3]K) {
	t.Helper()
	ix := persistenceIndex(t, keys)
	got := checkRoundTrip(t, ix)
	if got.Len() != 2 || got.Contains(keys[2]) || !got.Contains(keys[0]) || !got.Contains(keys[1]) {
		t.Fatalf("wrong keys after Load")
	}
	res, err := got.Search([]float32{1, 0}, 10, SearchOptions{Ef: 20})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range res {
		if r.Key == keys[2] {
			t.Fatal("deleted key returned after Load")
		}
	}
}

func TestPersistenceEveryKeyKind(t *testing.T) {
	checkPersistenceKeys(t, [3]string{"a", "", "gone"})
	checkPersistenceKeys(t, [3]namedString{"a", "b", "gone"})
	checkPersistenceKeys(t, [3]int{0, -1, 2})
	checkPersistenceKeys(t, [3]namedInt{1, 0, -1})
	checkPersistenceKeys(t, [3]int32{1, math.MinInt32, 3})
	checkPersistenceKeys(t, [3]namedInt32{1, 2, 3})
	checkPersistenceKeys(t, [3]int64{math.MaxInt64, math.MinInt64, 0})
	checkPersistenceKeys(t, [3]namedInt64{1, 2, 3})
	checkPersistenceKeys(t, [3]uint{0, 1, math.MaxUint})
	checkPersistenceKeys(t, [3]namedUint{1, 2, 3})
	checkPersistenceKeys(t, [3]uint32{0, math.MaxUint32, 7})
	checkPersistenceKeys(t, [3]namedUint32{1, 2, 3})
	checkPersistenceKeys(t, [3]uint64{math.MaxUint64, 0, 1})
	checkPersistenceKeys(t, [3]namedUint64{1, 2, 3})
	checkPersistenceKeys(t, [3][16]byte{{1}, {}, {15: 1}})
	checkPersistenceKeys(t, [3]namedUUID{{1}, {2}, {3}})
}

func TestPersistenceEmptyAndContinue(t *testing.T) {
	empty, err := New[int](3, Options{})
	if err != nil {
		t.Fatal(err)
	}
	got := checkRoundTrip(t, empty)
	if got.Len() != 0 || got.g.entry != noEntry {
		t.Fatal("empty index changed")
	}

	vecs := clustered(40, 8, 81)
	original := build(t, vecs[:20], Options{M: 4, EfConstruction: 30, Ef: 23, Seed: 17})
	loaded := checkRoundTrip(t, original)
	for i := 20; i < len(vecs); i++ {
		if err := original.Add(i, vecs[i]); err != nil {
			t.Fatal(err)
		}
		if err := loaded.Add(i, vecs[i]); err != nil {
			t.Fatal(err)
		}
	}
	if a, b := persistenceBytes(t, original), persistenceBytes(t, loaded); !bytes.Equal(a, b) {
		t.Fatal("Load then continue differs from an uninterrupted build")
	}
}

func TestLoadKeyConversion(t *testing.T) {
	ix, err := New[int64](2, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.Add(7, []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	data := persistenceBytes(t, ix)
	if _, err := Load[int32](bytes.NewReader(data)); err != nil {
		t.Fatal(err)
	}
	if _, err := Load[string](bytes.NewReader(data)); !errors.Is(err, ErrKeyType) {
		t.Fatalf("wrong key kind: %v", err)
	}
	if err := ix.Add(math.MaxInt64, []float32{0, 1}); err != nil {
		t.Fatal(err)
	}
	data = persistenceBytes(t, ix)
	if _, err := Load[int32](bytes.NewReader(data)); !errors.Is(err, ErrKeyType) {
		t.Fatalf("overflowing key: %v", err)
	}

	u, err := New[uint64](2, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if err := u.Add(math.MaxUint64, []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	if _, err := Load[uint32](bytes.NewReader(persistenceBytes(t, u))); !errors.Is(err, ErrKeyType) {
		t.Fatalf("unsigned overflow: %v", err)
	}
}

func TestLoadStopsAtTrailer(t *testing.T) {
	ix := persistenceIndex(t, [3]uint32{1, 2, 3})
	data := persistenceBytes(t, ix)
	r := bytes.NewReader(append(slices.Clone(data), 'x', 'y', 'z'))
	if _, err := Load[uint32](r); err != nil {
		t.Fatal(err)
	}
	if r.Len() != 3 {
		t.Fatalf("Load read %d trailing bytes", 3-r.Len())
	}
	path := filepath.Join(t.TempDir(), "trailing.hnsw")
	if err := os.WriteFile(path, append(slices.Clone(data), 0), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadFile[uint32](path); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("trailing file byte: %v", err)
	}
}

func TestLoadRejectsTruncationAndByteFlips(t *testing.T) {
	ix := persistenceIndex(t, [3]uint32{1, 2, 3})
	data := persistenceBytes(t, ix)
	for i := range data {
		_, err := Load[uint32](bytes.NewReader(data[:i]))
		if !errors.Is(err, ErrCorrupt) {
			t.Fatalf("prefix %d: %v, want ErrCorrupt", i, err)
		}
	}
	for i := range data {
		bad := slices.Clone(data)
		bad[i] ^= 1
		if _, err := Load[uint32](bytes.NewReader(bad)); err == nil {
			t.Fatalf("byte flip at %d was accepted", i)
		}
	}
}

func sectionPayload(t *testing.T, data []byte, tag string) []byte {
	t.Helper()
	for off := fileHeaderSize; off+12 <= len(data); {
		length := int(binary.LittleEndian.Uint64(data[off+4:]))
		start := off + 12
		if length < 0 || start+length+4 > len(data) {
			t.Fatal("invalid test file")
		}
		if string(data[off:start-8]) == tag {
			return data[start : start+length]
		}
		off = start + length + 4
	}
	t.Fatalf("section %q not found", tag)
	return nil
}

func TestLoadValidatesStructure(t *testing.T) {
	one, err := New[uint32](2, Options{M: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := one.Add(1, []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	data := persistenceBytes(t, one)
	bad := slices.Clone(data)
	adj := sectionPayload(t, bad, "ADJ0")
	binary.LittleEndian.PutUint32(adj, 1)
	binary.LittleEndian.PutUint32(adj[4:], 0)
	if _, err := load[uint32](bytes.NewReader(bad), true); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("self-loop: %v", err)
	}
	bad = slices.Clone(data)
	clear(sectionPayload(t, bad, "VECT"))
	if _, err := load[uint32](bytes.NewReader(bad), true); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("zero vector: %v", err)
	}

	two, err := New[uint32](2, Options{M: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := two.Add(1, []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := two.Add(2, []float32{0, 1}); err != nil {
		t.Fatal(err)
	}
	bad = persistenceBytes(t, two)
	vectors := sectionPayload(t, bad, "VECT")
	copy(vectors[:8], vectors[8:16])
	if _, err := load[uint32](bytes.NewReader(bad), true); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("duplicate vectors: %v", err)
	}
}

func TestPersistenceInvalidArgumentsAndErrors(t *testing.T) {
	var zero Index[int]
	if _, err := zero.WriteTo(io.Discard); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("zero Index WriteTo: %v", err)
	}
	ix, err := New[int](2, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ix.WriteTo(nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil writer: %v", err)
	}
	if _, err := Load[int](nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil reader: %v", err)
	}
	if _, err := Load[int](bytes.NewReader(nil)); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("empty input: %v", err)
	}
	if _, err := LoadFile[int](filepath.Join(t.TempDir(), "missing")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing file: %v", err)
	}

	boom := errors.New("reader boom")
	if _, err := Load[int](failingReader{boom}); !errors.Is(err, boom) || errors.Is(err, ErrCorrupt) {
		t.Fatalf("reader error: %v", err)
	}
	if err := ix.Add(1, []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	for _, limit := range []int{0, 1, 80, 100} {
		_, err := ix.WriteTo(&failingWriter{remaining: limit, err: boom})
		if !errors.Is(err, boom) {
			t.Fatalf("writer limit %d: %v", limit, err)
		}
	}

	data := persistenceBytes(t, ix)
	binary.LittleEndian.PutUint32(data[8:], 2)
	if _, err := Load[int](bytes.NewReader(data)); !errors.Is(err, ErrVersion) {
		t.Fatalf("version: %v", err)
	}
}

type failingReader struct{ err error }

func (r failingReader) Read([]byte) (int, error) { return 0, r.err }

type failingWriter struct {
	remaining int
	err       error
}

func (w *failingWriter) Write(p []byte) (int, error) {
	if w.remaining <= 0 {
		return 0, w.err
	}
	if len(p) > w.remaining {
		n := w.remaining
		w.remaining = 0
		return n, w.err
	}
	w.remaining -= len(p)
	return len(p), nil
}

func TestSaveFile(t *testing.T) {
	ix := persistenceIndex(t, [3]string{"a", "b", "gone"})
	dir := t.TempDir()
	path := filepath.Join(dir, "index.hnsw")
	if err := ix.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	got := checkRoundTrip(t, ix)
	loaded, err := LoadFile[string](path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(slices.Collect(loaded.Keys()), slices.Collect(got.Keys())) {
		t.Fatal("SaveFile then LoadFile changed keys")
	}
	if runtime.GOOS != "windows" {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode().Perm() != 0o600 {
			t.Fatalf("new file mode %o", info.Mode().Perm())
		}
		if err := os.Chmod(path, 0o640); err != nil {
			t.Fatal(err)
		}
		if err := ix.SaveFile(path); err != nil {
			t.Fatal(err)
		}
		info, err = os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode().Perm() != 0o640 {
			t.Fatalf("replacement mode %o", info.Mode().Perm())
		}
	}

	old := []byte("old complete file")
	if err := os.WriteFile(path, old, 0o600); err != nil {
		t.Fatal(err)
	}
	var zero Index[string]
	if err := zero.SaveFile(path); err == nil {
		t.Fatal("zero Index SaveFile succeeded")
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(after, old) {
		t.Fatal("failed save replaced the old file")
	}
}

func TestSaveFileSymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation needs extra privileges on Windows")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, "target.hnsw")
	link := filepath.Join(dir, "link.hnsw")
	ix := persistenceIndex(t, [3]int{1, 2, 3})
	if err := ix.SaveFile(target); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	if err := ix.Add(4, []float32{0, -1}); err != nil {
		t.Fatal(err)
	}
	if err := ix.SaveFile(link); err != nil {
		t.Fatal(err)
	}
	if info, err := os.Lstat(link); err != nil || info.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("symlink was replaced: %v, %v", info, err)
	}
	if _, err := LoadFile[int](target); err != nil {
		t.Fatal(err)
	}
}

func goldenString(t *testing.T) *Index[string] {
	return persistenceIndex(t, [3]string{"alpha", "alias", "gone"})
}

func goldenUint64(t *testing.T) *Index[uint64] {
	return persistenceIndex(t, [3]uint64{1, math.MaxUint64, 7})
}

func checkGolden[K Key](t *testing.T, path string, ix *Index[K]) {
	t.Helper()
	want := persistenceBytes(t, ix)
	if os.Getenv("HNSW_UPDATE_GOLDEN") != "" {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, want, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, want) {
		t.Fatal("golden file differs from WriteTo output")
	}
	got, err := LoadFile[K](path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Stats() != ix.Stats() {
		t.Fatalf("golden stats %+v, want %+v", got.Stats(), ix.Stats())
	}
}

func TestGoldenFiles(t *testing.T) {
	checkGolden(t, filepath.Join("testdata", "v1_string.hnsw"), goldenString(t))
	checkGolden(t, filepath.Join("testdata", "v1_uint64.hnsw"), goldenUint64(t))
}

func FuzzLoad(f *testing.F) {
	ix := persistenceIndex(f, [3]uint32{1, 2, 3})
	var buf bytes.Buffer
	if _, err := ix.WriteTo(&buf); err != nil {
		f.Fatal(err)
	}
	f.Add(buf.Bytes())
	f.Add([]byte{})
	f.Add([]byte("not an hnsw file"))
	f.Fuzz(func(t *testing.T, data []byte) {
		for _, skipCRC := range []bool{false, true} {
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			ix, err := load[uint32](bytes.NewReader(data), skipCRC)
			runtime.ReadMemStats(&after)
			used := after.TotalAlloc - before.TotalAlloc
			limit := uint64(len(data))*64 + 1<<20
			if used > limit {
				t.Fatalf("Load allocated %d bytes for %d input bytes, limit %d", used, len(data), limit)
			}
			if err != nil {
				continue
			}
			if err := ix.g.verify(); err != nil {
				t.Fatal(err)
			}
			q := make([]float32, ix.dim)
			q[0] = 1
			if _, err := ix.Search(q, 1, SearchOptions{}); err != nil {
				t.Fatal(err)
			}
			once := persistenceBytes(t, ix)
			again := checkRoundTrip(t, ix)
			if !bytes.Equal(once, persistenceBytes(t, again)) {
				t.Fatal("WriteTo then Load is not a fixpoint")
			}
		}
	})
}
