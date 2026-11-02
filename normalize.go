package hnsw

import (
	"fmt"
	"math"
)

// normTolerance is how far a squared norm may sit from 1 and keep its bits. Normalizing
// lands within about 1.2e-7, so normalizing twice changes nothing.
const normTolerance = 1e-6

// normalize copies src to dst, validates it and scales it to unit length. The lengths
// must match. dst is left unspecified on error.
//
// The result is bit-stable across GOARCH: float32 squares are exact in float64, so fusing
// the sum changes nothing, and the division cannot fuse.
func normalize(dst, src []float32) error {
	// Validate the copy, so a caller mutating src cannot slip an unchecked value through.
	copy(dst, src)
	var sum float64
	for _, x := range dst {
		y := float64(x)
		sum += y * y
	}
	if math.IsNaN(sum) || math.IsInf(sum, 0) {
		return fmt.Errorf("%w: NaN or Inf element", ErrInvalidVector)
	}
	if sum == 0 {
		return fmt.Errorf("%w: zero vector", ErrInvalidVector)
	}
	if math.Abs(sum-1) > normTolerance {
		norm := math.Sqrt(sum)
		for i, x := range dst {
			dst[i] = float32(float64(x) / norm)
		}
	}
	// Last, because division can underflow to -0. Merging compares bits, so -0 and +0
	// would otherwise make separate nodes at distance about 0.
	for i, x := range dst {
		if x == 0 {
			dst[i] = 0
		}
	}
	return nil
}
