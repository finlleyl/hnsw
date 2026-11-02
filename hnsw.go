package hnsw

import "errors"

// Errors returned by this package. Match them with errors.Is; they are wrapped with detail.
var (
	// ErrDimension means a vector's length differs from the index dimension.
	ErrDimension = errors.New("hnsw: wrong vector dimension")
	// ErrInvalidVector means a vector holds NaN or Inf, or is the zero vector.
	ErrInvalidVector = errors.New("hnsw: invalid vector")
	// ErrInvalidArgument means an option or argument is out of range or nil.
	ErrInvalidArgument = errors.New("hnsw: invalid argument")
	// ErrFull means the index ran out of node IDs or upper-layer records.
	ErrFull = errors.New("hnsw: index full")
	// ErrCorrupt means a file is truncated or fails validation.
	ErrCorrupt = errors.New("hnsw: corrupt file")
	// ErrVersion means a file has a format version this package cannot read.
	ErrVersion = errors.New("hnsw: unsupported file version")
	// ErrKeyType means a file's keys do not fit the requested key type.
	ErrKeyType = errors.New("hnsw: key type mismatch")
)
