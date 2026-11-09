//go:build !hnswnolock

package hnsw

import "sync"

type rwLock struct {
	sync.RWMutex
}
