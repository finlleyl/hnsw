package hnsw

// dot returns the dot product of a and b, which must have equal lengths. Sixteen
// accumulators were fastest on linux/amd64 in G4; eight remain faster on arm64.
func dot(a, b []float32) float32 {
	b = b[:len(a)]
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
