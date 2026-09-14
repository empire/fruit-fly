package connectome

import (
	"fmt"
	"os"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
)

// table holds selected columns of a feather file, read fully into memory.
type table struct {
	ints    map[string][]int64
	strings map[string][]string
}

// readFeather reads the named columns of a feather (Arrow IPC file) into Go slices.
// Null strings become "".
func readFeather(path string, intCols, stringCols []string) (*table, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r, err := ipc.NewFileReader(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	defer r.Close()

	t := &table{ints: map[string][]int64{}, strings: map[string][]string{}}
	index := func(name string) (int, error) {
		ids := r.Schema().FieldIndices(name)
		if len(ids) == 0 {
			return 0, fmt.Errorf("%s: no column %q", path, name)
		}
		return ids[0], nil
	}

	for i := range r.NumRecords() {
		batch, err := r.RecordBatchAt(i)
		if err != nil {
			return nil, fmt.Errorf("%s batch %d: %w", path, i, err)
		}
		for _, name := range intCols {
			col, err := index(name)
			if err != nil {
				batch.Release()
				return nil, err
			}
			ints, ok := batch.Column(col).(*array.Int64)
			if !ok {
				batch.Release()
				return nil, fmt.Errorf("%s: column %q is %s, want int64", path, name, batch.Column(col).DataType())
			}
			t.ints[name] = append(t.ints[name], ints.Int64Values()...)
		}
		for _, name := range stringCols {
			col, err := index(name)
			if err != nil {
				batch.Release()
				return nil, err
			}
			t.strings[name] = appendStrings(t.strings[name], batch.Column(col))
		}
		batch.Release()
	}
	return t, nil
}

func appendStrings(dst []string, col arrow.Array) []string {
	switch s := col.(type) {
	case *array.String:
		for i := range s.Len() {
			if s.IsNull(i) {
				dst = append(dst, "")
			} else {
				dst = append(dst, s.Value(i))
			}
		}
	case *array.LargeString:
		for i := range s.Len() {
			if s.IsNull(i) {
				dst = append(dst, "")
			} else {
				dst = append(dst, s.Value(i))
			}
		}
	default:
		panic(fmt.Sprintf("unsupported string column type %s", col.DataType()))
	}
	return dst
}
