// Package hnsw is an in-memory approximate nearest neighbour index for float32
// embeddings under cosine distance. All methods are safe for concurrent use.
//
// The concurrency contract is per key. After Add returns, Contains sees the key, and
// Search returns it subject to recall. After Delete returns, no later Search returns the
// key. One Search never returns the same key twice and never mixes two versions of a key.
// No multi-key snapshot is promised.
package hnsw
