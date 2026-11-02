package hnsw

// item is a node with its distance to the current query.
type item struct {
	dist float32
	id   uint32
}

// less orders items by (distance, ID), so pop order depends only on the set of items.
func (x item) less(y item) bool {
	return x.dist < y.dist || (x.dist == y.dist && x.id < y.id)
}

// minHeap pops the smallest item first. It allocates only when append grows it.
type minHeap []item

func (h *minHeap) push(x item) {
	s := append(*h, x)
	i := len(s) - 1
	for i > 0 {
		p := (i - 1) / 2
		if !x.less(s[p]) {
			break
		}
		s[i] = s[p]
		i = p
	}
	s[i] = x
	*h = s
}

func (h *minHeap) pop() item {
	s := *h
	top := s[0]
	n := len(s) - 1
	x := s[n]
	s = s[:n]
	*h = s
	if n == 0 {
		return top
	}
	i := 0
	for {
		c := 2*i + 1
		if c >= n {
			break
		}
		if c+1 < n && s[c+1].less(s[c]) {
			c++
		}
		if !s[c].less(x) {
			break
		}
		s[i] = s[c]
		i = c
	}
	s[i] = x
	return top
}

// maxHeap pops the largest item first. It allocates only when append grows it.
type maxHeap []item

func (h *maxHeap) push(x item) {
	s := append(*h, x)
	i := len(s) - 1
	for i > 0 {
		p := (i - 1) / 2
		if !s[p].less(x) {
			break
		}
		s[i] = s[p]
		i = p
	}
	s[i] = x
	*h = s
}

func (h *maxHeap) pop() item {
	s := *h
	top := s[0]
	n := len(s) - 1
	x := s[n]
	s = s[:n]
	*h = s
	if n == 0 {
		return top
	}
	i := 0
	for {
		c := 2*i + 1
		if c >= n {
			break
		}
		if c+1 < n && s[c].less(s[c+1]) {
			c++
		}
		if !x.less(s[c]) {
			break
		}
		s[i] = s[c]
		i = c
	}
	s[i] = x
	return top
}
