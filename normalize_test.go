package hnsw

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"testing"
)

// splitmix is a local generator, so golden values cannot move with the standard library.
type splitmix uint64

func (s *splitmix) next() uint64 {
	*s += 0x9e3779b97f4a7c15
	z := uint64(*s)
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// finite returns a float32 with uniform random bits, NaN and Inf excluded.
func (s *splitmix) finite() float32 {
	for {
		x := math.Float32frombits(uint32(s.next()))
		if !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) {
			return x
		}
	}
}

// signed returns a value in [-1, 1).
func (s *splitmix) signed() float32 {
	return float32(int32(s.next())) / (1 << 31)
}

// testVector draws one vector from a mix of shapes that reach both normalize branches,
// underflow, -0 and extreme exponents. It may return a zero vector only with vanishing
// probability; callers treat ErrInvalidVector as a test failure.
func testVector(s *splitmix, d int) []float32 {
	v := make([]float32, d)
	switch s.next() % 5 {
	case 0: // any exponent, so a few elements dominate and the rest underflow
		for i := range v {
			v[i] = s.finite()
		}
	case 1: // ordinary embedding shape
		for i := range v {
			v[i] = s.signed()
		}
	case 2: // tiny scale, including subnormals
		for i := range v {
			v[i] = s.signed() * 1e-40
		}
	case 3: // sparse, with -0 in the gaps
		v[s.next()%uint64(d)] = 1 + s.signed()/2
		for i := range v {
			if v[i] == 0 && s.next()%2 == 0 {
				v[i] = float32(math.Copysign(0, -1))
			}
		}
	case 4: // already unit, then nudged by an ulp, so the bits-kept branch runs
		for i := range v {
			v[i] = s.signed()
		}
		if err := normalize(v, v); err != nil {
			panic(err)
		}
		i := s.next() % uint64(d)
		v[i] = math.Float32frombits(math.Float32bits(v[i]) + 1)
	}
	return v
}

func squaredNorm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return sum
}

func sameBits(a, b []float32) bool {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return len(a) == len(b)
}

var testDims = []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 128, 384, 768, 1536, 3072}

func TestNormalizeProperties(t *testing.T) {
	s := splitmix(1)
	check := func(src []float32) {
		t.Helper()
		orig := append([]float32(nil), src...)
		once := make([]float32, len(src))
		if err := normalize(once, src); err != nil {
			t.Fatalf("d=%d: %v", len(src), err)
		}
		if !sameBits(src, orig) {
			t.Fatalf("d=%d: src was modified", len(src))
		}
		if sum := squaredNorm(once); math.Abs(sum-1) > normTolerance {
			t.Fatalf("d=%d: squared norm %v", len(src), sum)
		}
		for i, x := range once {
			if x == 0 && math.Signbit(float64(x)) {
				t.Fatalf("d=%d: -0 at %d", len(src), i)
			}
		}
		twice := make([]float32, len(src))
		if err := normalize(twice, once); err != nil {
			t.Fatalf("d=%d: second pass: %v", len(src), err)
		}
		if !sameBits(once, twice) {
			t.Fatalf("d=%d: not idempotent", len(src))
		}
	}
	for range 2000 {
		check(testVector(&s, testDims[s.next()%uint64(len(testDims))]))
	}
	check(testVector(&s, 65536))
}

func TestNormalizeRejects(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	negZero := float32(math.Copysign(0, -1))
	for name, v := range map[string][]float32{
		"NaN":         {1, nan, 0},
		"+Inf":        {1, inf, 0},
		"-Inf":        {1, -inf, 0},
		"Inf and NaN": {inf, nan},
		"zero":        {0, 0, 0},
		"-0":          {negZero, negZero},
		"empty":       {},
	} {
		if err := normalize(make([]float32, len(v)), v); !errors.Is(err, ErrInvalidVector) {
			t.Errorf("%s: got %v, want ErrInvalidVector", name, err)
		}
	}
}

func TestNormalizeKeepsNearUnitBits(t *testing.T) {
	// Adjacent floats straddling both sides of the 1e-6 tolerance on the squared norm.
	for _, tc := range []struct {
		bits uint32
		keep bool
	}{
		{0x3f800004, true},  // 1+9.54e-7
		{0x3f800005, false}, // 1+1.19e-6
		{0x3f7ffff8, true},  // 1-9.54e-7
		{0x3f7ffff7, false}, // 1-1.07e-6
	} {
		in := []float32{math.Float32frombits(tc.bits), 0}
		want := in
		if !tc.keep {
			want = []float32{1, 0}
		}
		out := make([]float32, 2)
		if err := normalize(out, in); err != nil {
			t.Fatal(err)
		}
		if !sameBits(out, want) {
			t.Errorf("%#x: got %#x", tc.bits, math.Float32bits(out[0]))
		}
	}
}

func TestNormalizeNegativeZero(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	a, b := make([]float32, 2), make([]float32, 2)
	if err := normalize(a, []float32{1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := normalize(b, []float32{1, negZero}); err != nil {
		t.Fatal(err)
	}
	if !sameBits(a, b) {
		t.Fatalf("[1,0] and [1,-0] differ: %x %x", math.Float32bits(a[1]), math.Float32bits(b[1]))
	}
	// A negative element that underflows during division becomes -0 before the last pass.
	c := make([]float32, 2)
	if err := normalize(c, []float32{math.MaxFloat32, -math.SmallestNonzeroFloat32}); err != nil {
		t.Fatal(err)
	}
	if !sameBits(c, []float32{1, 0}) {
		t.Fatalf("underflow: %x %x", math.Float32bits(c[0]), math.Float32bits(c[1]))
	}
}

func TestNormalizeGoldenBits(t *testing.T) {
	for _, tc := range []struct {
		in   []float32
		want []uint32
	}{
		{[]float32{3, 4}, []uint32{0x3f19999a, 0x3f4ccccd}},
		{[]float32{-3, 4}, []uint32{0xbf19999a, 0x3f4ccccd}},
		{[]float32{1, 1}, []uint32{0x3f3504f3, 0x3f3504f3}},
		{[]float32{math.MaxFloat32, math.MaxFloat32}, []uint32{0x3f3504f3, 0x3f3504f3}},
		{[]float32{math.SmallestNonzeroFloat32, 0}, []uint32{0x3f800000, 0}},
	} {
		out := make([]float32, len(tc.in))
		if err := normalize(out, tc.in); err != nil {
			t.Fatal(err)
		}
		for i := range out {
			if got := math.Float32bits(out[i]); got != tc.want[i] {
				t.Errorf("%v[%d]: got %#x, want %#x", tc.in, i, got, tc.want[i])
			}
		}
	}

	// Every CI architecture must produce these bytes for a stream covering both branches.
	const want = "603972f46c1cb2e8b7e67b555c62560c5e9f310c35f3e35843275558c95c2928"
	s := splitmix(42)
	h := sha256.New()
	for range 5000 {
		v := testVector(&s, testDims[s.next()%uint64(len(testDims))])
		if err := normalize(v, v); err != nil {
			t.Fatal(err)
		}
		for _, x := range v {
			if err := binary.Write(h, binary.LittleEndian, math.Float32bits(x)); err != nil {
				t.Fatal(err)
			}
		}
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		t.Errorf("stream hash %s, want %s", got, want)
	}
}
