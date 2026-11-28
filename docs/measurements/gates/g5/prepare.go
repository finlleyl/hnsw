package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/parquet-go/parquet-go"
)

const datasetURL = "https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M/resolve/b2f301648eda4c5dbe659235589c86ffec702581/data"

type parquetRow struct {
	Embedding []float64 `parquet:"text-embedding-3-large-3072-embedding,list"`
}

func download() error {
	if *shards < 1 || *shards > 63 {
		return fmt.Errorf("shards %d outside 1..63", *shards)
	}
	dir := filepath.Join(*dataDir, "parquet")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	for i := range *shards {
		name := fmt.Sprintf("train-%05d-of-00063.parquet", i)
		path := filepath.Join(dir, name)
		if st, err := os.Stat(path); err == nil && st.Size() > 0 {
			log.Printf("%s already downloaded (%d bytes)", name, st.Size())
			continue
		}
		url := datasetURL + "/" + name
		if err := downloadFile(url, path); err != nil {
			return err
		}
	}
	return nil
}

func downloadFile(url, path string) error {
	var err error
	for attempt := range 3 {
		if attempt > 0 {
			time.Sleep(time.Duration(attempt) * 5 * time.Second)
		}
		log.Printf("downloading %s", url)
		var resp *http.Response
		resp, err = http.Get(url)
		if err == nil && resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			err = fmt.Errorf("%s: %s", url, resp.Status)
		}
		if err != nil {
			continue
		}
		tmp := path + ".part"
		var f *os.File
		f, err = os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
		if err == nil {
			_, err = io.Copy(f, resp.Body)
		}
		resp.Body.Close()
		if f != nil {
			if closeErr := f.Close(); err == nil {
				err = closeErr
			}
		}
		if err == nil {
			err = os.Rename(tmp, path)
		}
		if err == nil {
			return nil
		}
		log.Printf("download failed: %v", err)
	}
	return err
}

type rawOutput struct {
	dim  int
	path string
	tmp  string
	file *os.File
	buf  *bufio.Writer
	vec  []float32
	bin  []byte
}

func prepare() error {
	if err := os.MkdirAll(*dataDir, 0o700); err != nil {
		return err
	}
	outputs := make([]*rawOutput, 0, 3)
	for _, d := range []int{768, 1536, 3072} {
		out, err := openRawOutput(d)
		if err != nil {
			return err
		}
		if out != nil {
			outputs = append(outputs, out)
		}
	}
	if len(outputs) == 0 {
		log.Print("normalized files already exist")
		return nil
	}
	defer func() {
		for _, out := range outputs {
			out.file.Close()
			os.Remove(out.tmp)
		}
	}()
	seen := 0
	for shard := 0; shard < *shards && seen < *rows; shard++ {
		name := fmt.Sprintf("train-%05d-of-00063.parquet", shard)
		path := filepath.Join(*dataDir, "parquet", name)
		if err := prepareShard(path, outputs, &seen); err != nil {
			return err
		}
		log.Printf("prepared %d/%d rows", seen, *rows)
	}
	if seen != *rows {
		return fmt.Errorf("read %d rows, want %d", seen, *rows)
	}
	for _, out := range outputs {
		if err := out.buf.Flush(); err != nil {
			return err
		}
		if err := out.file.Close(); err != nil {
			return err
		}
		if err := os.Rename(out.tmp, out.path); err != nil {
			return err
		}
	}
	return nil
}

func openRawOutput(d int) (*rawOutput, error) {
	path := rawPath(d)
	if st, err := os.Stat(path); err == nil {
		want := int64(*rows) * int64(d) * 4
		if st.Size() != want {
			return nil, fmt.Errorf("%s is %d bytes, want %d", path, st.Size(), want)
		}
		return nil, nil
	}
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	return &rawOutput{
		dim: d, path: path, tmp: tmp, file: f,
		buf: bufio.NewWriterSize(f, 1<<20), vec: make([]float32, d), bin: make([]byte, 0, d*4),
	}, nil
}

func prepareShard(path string, outputs []*rawOutput, seen *int) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	r := parquet.NewGenericReader[parquetRow](f)
	defer r.Close()
	batch := make([]parquet.Row, 1)
	for *seen < *rows {
		n, err := r.ReadRows(batch)
		for i := range n {
			row := batch[i]
			if len(row) != 3072 {
				return fmt.Errorf("%s row %d has %d dimensions; file schema:\n%s\nreader schema:\n%s", path, *seen, len(row), r.File().Schema(), r.Schema())
			}
			for _, out := range outputs {
				for j := range out.vec {
					out.vec[j] = float32(row[j].Double())
				}
				if err := normalize(out.vec); err != nil {
					return fmt.Errorf("%s row %d: %w", path, *seen, err)
				}
				var appendErr error
				out.bin, appendErr = binary.Append(out.bin[:0], binary.LittleEndian, out.vec)
				if appendErr != nil {
					return appendErr
				}
				if _, err := out.buf.Write(out.bin); err != nil {
					return err
				}
			}
			clear(row)
			(*seen)++
			if *seen == *rows {
				break
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return fmt.Errorf("%s: reader made no progress", path)
		}
	}
	return nil
}
