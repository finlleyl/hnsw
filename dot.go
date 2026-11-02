package hnsw

// dot returns the dot product of a and b, which must have equal lengths. Separate
// accumulators let the CPU run the eight lanes in parallel.
func dot(a, b []float32) float32 {
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
