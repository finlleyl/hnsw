//go:build hnswnolock

package hnsw

type rwLock struct{}

func (rwLock) Lock()    {}
func (rwLock) Unlock()  {}
func (rwLock) RLock()   {}
func (rwLock) RUnlock() {}
