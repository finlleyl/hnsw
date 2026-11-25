package hnsw

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
)

const (
	fileHeaderSize  = 80
	fileTrailerSize = 16
)

var (
	fileMagic   = [8]byte{0x89, 'H', 'N', 'S', 'W', 0x0d, 0x0a, 0x1a}
	fileTrailer = [8]byte{'H', 'N', 'S', 'W', 'E', 'N', 'D', '\n'}
	crcTable    = crc32.MakeTable(crc32.Castagnoli)
)

// WriteTo writes one consistent snapshot, tombstones included. Searches continue and
// writers block. w must not call Add, Delete, Compact or WriteTo on this index.
func (ix *Index[K]) WriteTo(w io.Writer) (int64, error) {
	if w == nil {
		return 0, fmt.Errorf("%w: nil writer", ErrInvalidArgument)
	}
	ix.wmu.Lock()
	defer ix.wmu.Unlock()
	g := ix.g
	if g == nil {
		return 0, fmt.Errorf("%w: zero Index", ErrInvalidArgument)
	}
	kind := keyKind(reflect.TypeFor[K]())
	if kind == 0 {
		return 0, fmt.Errorf("%w: unsupported key type", ErrKeyType)
	}
	aliases := make([]uint32, 0, len(g.alias))
	for id := range g.alias {
		aliases = append(aliases, id)
	}
	slices.Sort(aliases)
	keysLen, err := keySectionLength(g, kind)
	if err != nil {
		return 0, err
	}
	aliasLen, err := aliasSectionLength(g, aliases, kind)
	if err != nil {
		return 0, err
	}

	actual := &countingWriter{w: w}
	fw := formatWriter{w: bufio.NewWriterSize(actual, 64<<10), actual: actual}
	writeHeader(&fw, g, kind, ix.efc, ix.ef, ix.seed, len(aliases))
	fw.beginSection("VECT", uint64(g.n)*uint64(g.dim)*4)
	for _, chunk := range g.vecs {
		writeFloat32s(&fw, chunk)
	}
	fw.endSection()
	fw.beginSection("LEVL", uint64(len(g.level)))
	fw.sectionData(g.level)
	fw.endSection()
	fw.beginSection("DEAD", uint64(len(g.dead))*8)
	writeUint64s(&fw, g.dead)
	fw.endSection()
	fw.beginSection("ADJ0", uint64(len(g.adj0))*4)
	writeUint32s(&fw, g.adj0)
	fw.endSection()
	fw.beginSection("UPPR", uint64(len(g.up))*4)
	writeUint32s(&fw, g.up)
	fw.endSection()
	fw.beginSection("KEYS", keysLen)
	for id := uint32(0); id < g.n; id++ {
		if !g.isDead(id) {
			writeKey(&fw, g.keys[id], kind)
		}
	}
	fw.endSection()
	fw.beginSection("ALIS", aliasLen)
	for _, id := range aliases {
		as := g.alias[id]
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], id)
		fw.sectionData(b[:])
		binary.LittleEndian.PutUint32(b[:], uint32(len(as)))
		fw.sectionData(b[:])
		for _, key := range as {
			writeKey(&fw, key, kind)
		}
	}
	fw.endSection()
	return fw.finish()
}

// Load returns a new, fully validated index. Reading stops exactly at the trailer.
func Load[K Key](r io.Reader) (*Index[K], error) {
	return load[K](r, false)
}

func load[K Key](r io.Reader, skipCRC bool) (*Index[K], error) {
	if r == nil {
		return nil, fmt.Errorf("%w: nil reader", ErrInvalidArgument)
	}
	cr := countingReader{r: r}
	var header [fileHeaderSize]byte
	if err := readFull(&cr, header[:]); err != nil {
		return nil, err
	}
	if !slices.Equal(header[:8], fileMagic[:]) {
		return nil, corrupt("bad magic")
	}
	if version := binary.LittleEndian.Uint32(header[8:]); version != 1 {
		return nil, fmt.Errorf("%w: %d", ErrVersion, version)
	}
	kind := header[13]
	if kind != keyKind(reflect.TypeFor[K]()) {
		return nil, fmt.Errorf("%w: file key kind %d", ErrKeyType, kind)
	}
	if !skipCRC && crc32.Checksum(header[:76], crcTable) != binary.LittleEndian.Uint32(header[76:]) {
		return nil, corrupt("header checksum")
	}
	if header[12] != 1 || header[15] != 0 || binary.LittleEndian.Uint32(header[72:]) != 0 {
		return nil, corrupt("unsupported metric, reserved byte or flags")
	}
	dim := int(binary.LittleEndian.Uint32(header[16:]))
	m := int(binary.LittleEndian.Uint32(header[20:]))
	efc := int(binary.LittleEndian.Uint32(header[24:]))
	ef := int(binary.LittleEndian.Uint32(header[28:]))
	ix, err := New[K](dim, Options{M: m, EfConstruction: efc, Ef: ef, Seed: binary.LittleEndian.Uint64(header[32:])})
	if err != nil {
		return nil, corrupt("invalid header options: %v", err)
	}
	g := ix.g
	g.n = binary.LittleEndian.Uint32(header[40:])
	g.live = binary.LittleEndian.Uint32(header[44:])
	g.deadN = binary.LittleEndian.Uint32(header[48:])
	g.entry = binary.LittleEndian.Uint32(header[52:])
	g.top = header[14]
	upperWords := binary.LittleEndian.Uint64(header[56:])
	aliasCount := binary.LittleEndian.Uint64(header[64:])
	if g.n > maxNodes || g.deadN > g.n || g.live != g.n-g.deadN || g.top > maxLevel || aliasCount > uint64(g.live) {
		return nil, corrupt("invalid header counts")
	}
	if (g.n == 0) != (g.entry == noEntry) || g.n > 0 && g.entry >= g.n {
		return nil, corrupt("invalid entry %d for %d nodes", g.entry, g.n)
	}
	if uint64(g.n) > uint64(^uint(0)>>1) {
		return nil, corrupt("%d nodes do not fit this platform", g.n)
	}

	vectors, err := readFixedSection(&cr, "VECT", uint64(g.n)*uint64(dim)*4, skipCRC)
	if err != nil {
		return nil, err
	}
	g.vecs = decodeVectors(vectors, g.n, dim)
	levels, err := readFixedSection(&cr, "LEVL", uint64(g.n), skipCRC)
	if err != nil {
		return nil, err
	}
	g.level = slices.Clone(levels)
	g.upOff = make([]uint32, len(g.level))
	var records uint64
	for id, lv := range g.level {
		if lv > maxLevel || records > maxRecords {
			return nil, corrupt("invalid level or upper record count")
		}
		g.upOff[id] = uint32(records)
		records += uint64(lv)
	}
	if records > maxRecords || records*uint64(m+1) != upperWords {
		return nil, corrupt("upper word count %d, want %d records", upperWords, records)
	}
	dead, err := readFixedSection(&cr, "DEAD", (uint64(g.n)+63)/64*8, skipCRC)
	if err != nil {
		return nil, err
	}
	g.dead = decodeUint64s(dead)
	adj0, err := readFixedSection(&cr, "ADJ0", uint64(g.n)*uint64(2*m+1)*4, skipCRC)
	if err != nil {
		return nil, err
	}
	g.adj0 = decodeUint32s(adj0)
	upper, err := readFixedSection(&cr, "UPPR", upperWords*4, skipCRC)
	if err != nil {
		return nil, err
	}
	g.up = decodeUint32s(upper)
	keys, err := readKeysSection(&cr, kind, g.live, skipCRC)
	if err != nil {
		return nil, err
	}
	if err := decodePrimaryKeys(g, keys, kind); err != nil {
		return nil, err
	}
	aliases, err := readSection(&cr, "ALIS", skipCRC)
	if err != nil {
		return nil, err
	}
	if err := decodeAliases(g, aliases, kind, aliasCount); err != nil {
		return nil, err
	}
	var trailer [fileTrailerSize]byte
	if err := readFull(&cr, trailer[:]); err != nil {
		return nil, err
	}
	if !slices.Equal(trailer[:8], fileTrailer[:]) || binary.LittleEndian.Uint64(trailer[8:]) != cr.n {
		return nil, corrupt("bad trailer or file size")
	}
	g.byVec = make(map[uint64]uint32, int(g.n))
	for id := uint32(0); id < g.n; id++ {
		found, slot := g.findVec(g.vec(id))
		if found != noEntry {
			return nil, corrupt("nodes %d and %d have identical vectors", found, id)
		}
		g.byVec[slot] = id
	}
	if err := g.verify(); err != nil {
		return nil, err
	}
	return ix, nil
}

// LoadFile loads path and rejects trailing bytes. A missing file wraps fs.ErrNotExist.
func LoadFile[K Key](path string) (*Index[K], error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	ix, err := Load[K](f)
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	var extra [1]byte
	n, readErr := f.Read(extra[:])
	closeErr := f.Close()
	if n != 0 || readErr == nil {
		return nil, corrupt("trailing bytes")
	}
	if readErr != io.EOF {
		return nil, readErr
	}
	if closeErr != nil {
		return nil, closeErr
	}
	return ix, nil
}

// SaveFile atomically replaces path with a consistent snapshot.
func (ix *Index[K]) SaveFile(path string) error {
	target := path
	mode := os.FileMode(0o600)
	info, err := os.Lstat(path)
	switch {
	case err == nil:
		if info.Mode()&os.ModeSymlink != 0 {
			target, err = filepath.EvalSymlinks(path)
			if err != nil {
				return err
			}
			info, err = os.Stat(target)
			if err != nil {
				return err
			}
		}
		mode = info.Mode().Perm()
	case !errors.Is(err, os.ErrNotExist):
		return err
	}
	dir := filepath.Dir(target)
	f, err := os.CreateTemp(dir, "."+filepath.Base(target)+".tmp-")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if err := f.Chmod(mode); err != nil {
		_ = f.Close()
		return err
	}
	if _, err := ix.WriteTo(f); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, target); err != nil {
		return err
	}
	if runtime.GOOS == "windows" {
		return nil
	}
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	if err := d.Sync(); err != nil {
		_ = d.Close()
		return err
	}
	return d.Close()
}

func corrupt(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrCorrupt, fmt.Sprintf(format, args...))
}

type countingReader struct {
	r io.Reader
	n uint64
}

func (r *countingReader) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	r.n += uint64(n)
	return n, err
}

type countingWriter struct {
	w io.Writer
	n int64
}

func (w *countingWriter) Write(p []byte) (int, error) {
	n, err := w.w.Write(p)
	w.n += int64(n)
	return n, err
}

type formatWriter struct {
	w      *bufio.Writer
	actual *countingWriter
	off    uint64
	crc    uint32
	err    error
}

func (w *formatWriter) write(p []byte) {
	if w.err != nil {
		return
	}
	n, err := w.w.Write(p)
	w.off += uint64(n)
	w.err = err
}

func writeHeader[K Key](w *formatWriter, g *graph[K], kind byte, efc, ef int, seed uint64, aliasCount int) {
	var b [fileHeaderSize]byte
	copy(b[:], fileMagic[:])
	binary.LittleEndian.PutUint32(b[8:], 1)
	b[12] = 1
	b[13] = kind
	b[14] = g.top
	binary.LittleEndian.PutUint32(b[16:], uint32(g.dim))
	binary.LittleEndian.PutUint32(b[20:], uint32(g.m))
	binary.LittleEndian.PutUint32(b[24:], uint32(efc))
	binary.LittleEndian.PutUint32(b[28:], uint32(ef))
	binary.LittleEndian.PutUint64(b[32:], seed)
	binary.LittleEndian.PutUint32(b[40:], g.n)
	binary.LittleEndian.PutUint32(b[44:], g.live)
	binary.LittleEndian.PutUint32(b[48:], g.deadN)
	binary.LittleEndian.PutUint32(b[52:], g.entry)
	binary.LittleEndian.PutUint64(b[56:], uint64(len(g.up)))
	binary.LittleEndian.PutUint64(b[64:], uint64(aliasCount))
	binary.LittleEndian.PutUint32(b[76:], crc32.Checksum(b[:76], crcTable))
	w.write(b[:])
}

func (w *formatWriter) beginSection(tag string, length uint64) {
	var b [12]byte
	copy(b[:4], tag)
	binary.LittleEndian.PutUint64(b[4:], length)
	w.write(b[:])
	w.crc = 0
}

func (w *formatWriter) sectionData(p []byte) {
	w.write(p)
	w.crc = crc32.Update(w.crc, crcTable, p)
}

func (w *formatWriter) endSection() {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], w.crc)
	w.write(b[:])
}

func (w *formatWriter) finish() (int64, error) {
	if w.err == nil {
		var b [fileTrailerSize]byte
		copy(b[:], fileTrailer[:])
		binary.LittleEndian.PutUint64(b[8:], w.off+fileTrailerSize)
		w.write(b[:])
	}
	if err := w.w.Flush(); w.err == nil {
		w.err = err
	}
	return w.actual.n, w.err
}

func writeFloat32s(w *formatWriter, values []float32) {
	var buf [32 * 1024]byte
	for len(values) > 0 {
		n := min(len(values), len(buf)/4)
		for i, x := range values[:n] {
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(x))
		}
		w.sectionData(buf[:4*n])
		values = values[n:]
	}
}

func writeUint32s(w *formatWriter, values []uint32) {
	var buf [32 * 1024]byte
	for len(values) > 0 {
		n := min(len(values), len(buf)/4)
		for i, x := range values[:n] {
			binary.LittleEndian.PutUint32(buf[4*i:], x)
		}
		w.sectionData(buf[:4*n])
		values = values[n:]
	}
}

func writeUint64s(w *formatWriter, values []uint64) {
	var buf [32 * 1024]byte
	for len(values) > 0 {
		n := min(len(values), len(buf)/8)
		for i, x := range values[:n] {
			binary.LittleEndian.PutUint64(buf[8*i:], x)
		}
		w.sectionData(buf[:8*n])
		values = values[n:]
	}
}

func keyKind(t reflect.Type) byte {
	switch t.Kind() {
	case reflect.String:
		return 1
	case reflect.Int, reflect.Int32, reflect.Int64:
		return 2
	case reflect.Uint, reflect.Uint32, reflect.Uint64:
		return 3
	case reflect.Array:
		if t.Len() == 16 && t.Elem().Kind() == reflect.Uint8 {
			return 4
		}
	}
	return 0
}

func keyValue[K Key](key K) reflect.Value {
	return reflect.ValueOf(&key).Elem()
}

func encodedKeySize[K Key](key K, kind byte) uint64 {
	if kind == 1 {
		return 4 + uint64(keyValue(key).Len())
	}
	if kind == 4 {
		return 16
	}
	return 8
}

func addSectionSize(total *uint64, size uint64) error {
	if math.MaxUint64-*total < size {
		return corrupt("key section is too large")
	}
	*total += size
	return nil
}

func keySectionLength[K Key](g *graph[K], kind byte) (uint64, error) {
	var total uint64
	for id := uint32(0); id < g.n; id++ {
		if !g.isDead(id) {
			if err := addSectionSize(&total, encodedKeySize(g.keys[id], kind)); err != nil {
				return 0, err
			}
		}
	}
	return total, nil
}

func aliasSectionLength[K Key](g *graph[K], aliases []uint32, kind byte) (uint64, error) {
	var total uint64
	for _, id := range aliases {
		total += 8
		for _, key := range g.alias[id] {
			if err := addSectionSize(&total, encodedKeySize(key, kind)); err != nil {
				return 0, err
			}
		}
	}
	return total, nil
}

func writeKey[K Key](w *formatWriter, key K, kind byte) {
	v := keyValue(key)
	switch kind {
	case 1:
		var b [4]byte
		s := v.String()
		binary.LittleEndian.PutUint32(b[:], uint32(len(s)))
		w.sectionData(b[:])
		w.sectionData([]byte(s))
	case 2:
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], uint64(v.Int()))
		w.sectionData(b[:])
	case 3:
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], v.Uint())
		w.sectionData(b[:])
	case 4:
		var b [16]byte
		for i := range b {
			b[i] = byte(v.Index(i).Uint())
		}
		w.sectionData(b[:])
	}
}

func readFull(r io.Reader, p []byte) error {
	_, err := io.ReadFull(r, p)
	if err == nil {
		return nil
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return corrupt("truncated input")
	}
	return err
}

func readSection(r io.Reader, tag string, skipCRC bool) ([]byte, error) {
	var header [12]byte
	if err := readFull(r, header[:]); err != nil {
		return nil, err
	}
	if string(header[:4]) != tag {
		return nil, corrupt("section tag %q, want %q", header[:4], tag)
	}
	length := binary.LittleEndian.Uint64(header[4:])
	if length > uint64(^uint(0)>>1) {
		return nil, corrupt("section %q is too large", tag)
	}
	lr := &io.LimitedReader{R: r, N: int64(length)}
	var payload []byte
	for lr.N > 0 {
		n := min(lr.N, int64(32*1024))
		old := len(payload)
		payload = append(payload, make([]byte, int(n))...)
		if err := readFull(lr, payload[old:]); err != nil {
			return nil, err
		}
	}
	var checksum [4]byte
	if err := readFull(r, checksum[:]); err != nil {
		return nil, err
	}
	if !skipCRC && crc32.Checksum(payload, crcTable) != binary.LittleEndian.Uint32(checksum[:]) {
		return nil, corrupt("section %q checksum", tag)
	}
	return payload, nil
}

func readFixedSection(r io.Reader, tag string, length uint64, skipCRC bool) ([]byte, error) {
	p, err := readSection(r, tag, skipCRC)
	if err != nil {
		return nil, err
	}
	if uint64(len(p)) != length {
		return nil, corrupt("section %q holds %d bytes, want %d", tag, len(p), length)
	}
	return p, nil
}

func readKeysSection(r io.Reader, kind byte, live uint32, skipCRC bool) ([]byte, error) {
	if kind == 1 {
		return readSection(r, "KEYS", skipCRC)
	}
	size := uint64(8)
	if kind == 4 {
		size = 16
	}
	return readFixedSection(r, "KEYS", uint64(live)*size, skipCRC)
}

func decodeVectors(data []byte, n uint32, dim int) [][]float32 {
	out := make([][]float32, (int(n)+chunkSize-1)/chunkSize)
	off := 0
	for c := range out {
		count := min(int(n)-c*chunkSize, chunkSize)
		chunk := make([]float32, count*dim)
		for i := range chunk {
			chunk[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
			off += 4
		}
		out[c] = chunk
	}
	return out
}

func decodeUint32s(data []byte) []uint32 {
	out := make([]uint32, len(data)/4)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(data[4*i:])
	}
	return out
}

func decodeUint64s(data []byte) []uint64 {
	out := make([]uint64, len(data)/8)
	for i := range out {
		out[i] = binary.LittleEndian.Uint64(data[8*i:])
	}
	return out
}

func takeBytes(data []byte, off *int, size uint64) ([]byte, error) {
	if size > uint64(len(data)-*off) {
		return nil, corrupt("truncated key section")
	}
	start := *off
	*off += int(size)
	return data[start:*off], nil
}

func readKey[K Key](data []byte, off *int, kind byte) (K, error) {
	var zero K
	t := reflect.TypeFor[K]()
	v := reflect.New(t).Elem()
	switch kind {
	case 1:
		b, err := takeBytes(data, off, 4)
		if err != nil {
			return zero, err
		}
		s, err := takeBytes(data, off, uint64(binary.LittleEndian.Uint32(b)))
		if err != nil {
			return zero, err
		}
		v.SetString(string(s))
	case 2:
		b, err := takeBytes(data, off, 8)
		if err != nil {
			return zero, err
		}
		x := int64(binary.LittleEndian.Uint64(b))
		if v.OverflowInt(x) {
			return zero, fmt.Errorf("%w: %d does not fit %v", ErrKeyType, x, t)
		}
		v.SetInt(x)
	case 3:
		b, err := takeBytes(data, off, 8)
		if err != nil {
			return zero, err
		}
		x := binary.LittleEndian.Uint64(b)
		if v.OverflowUint(x) {
			return zero, fmt.Errorf("%w: %d does not fit %v", ErrKeyType, x, t)
		}
		v.SetUint(x)
	case 4:
		b, err := takeBytes(data, off, 16)
		if err != nil {
			return zero, err
		}
		for i, x := range b {
			v.Index(i).SetUint(uint64(x))
		}
	}
	return v.Interface().(K), nil
}

func decodePrimaryKeys[K Key](g *graph[K], data []byte, kind byte) error {
	off := 0
	var zero K
	for id := uint32(0); id < g.n; id++ {
		if g.isDead(id) {
			g.keys = append(g.keys, zero)
			continue
		}
		key, err := readKey[K](data, &off, kind)
		if err != nil {
			return err
		}
		if _, exists := g.byKey[key]; exists {
			return corrupt("duplicate key on node %d", id)
		}
		g.keys = append(g.keys, key)
		g.byKey[key] = id
	}
	if off != len(data) {
		return corrupt("%d unused primary key bytes", len(data)-off)
	}
	return nil
}

func decodeAliases[K Key](g *graph[K], data []byte, kind byte, count uint64) error {
	off := 0
	var previous uint32
	for i := uint64(0); i < count; i++ {
		b, err := takeBytes(data, &off, 8)
		if err != nil {
			return err
		}
		id := binary.LittleEndian.Uint32(b)
		n := binary.LittleEndian.Uint32(b[4:])
		if id >= g.n || g.isDead(id) || n == 0 || i > 0 && id <= previous {
			return corrupt("invalid alias entry for node %d", id)
		}
		previous = id
		for range n {
			key, err := readKey[K](data, &off, kind)
			if err != nil {
				return err
			}
			if _, exists := g.byKey[key]; exists {
				return corrupt("duplicate alias on node %d", id)
			}
			g.alias[id] = append(g.alias[id], key)
			g.byKey[key] = id
		}
	}
	if off != len(data) {
		return corrupt("%d unused alias bytes", len(data)-off)
	}
	return nil
}
