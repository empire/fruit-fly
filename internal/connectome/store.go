package connectome

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const graphMagic = "FLYGRAPH1"

// Save writes graph.bin (little-endian binary) and meta.json.
func Save(dir string, g *Graph, meta *Meta) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.Create(filepath.Join(dir, "graph.bin"))
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	put := func(v any) {
		if err == nil {
			err = binary.Write(w, binary.LittleEndian, v)
		}
	}
	putSlice := func(v any, n int) {
		put(int64(n))
		put(v)
	}
	put([]byte(graphMagic))
	put(int64(g.N))
	putSlice(g.Start, len(g.Start))
	putSlice(g.Target, len(g.Target))
	putSlice(g.Weight, len(g.Weight))
	putSlice(g.InDegree, len(g.InDegree))
	putSlice(g.Region, len(g.Region))
	for _, groups := range [][cells][]int32{g.OwnSensors, g.OppSensors} {
		for _, s := range groups {
			putSlice(s, len(s))
		}
	}
	putSlice(g.Readout, len(g.Readout))
	if err == nil {
		err = w.Flush()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}

	js, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "meta.json"), js, 0o644)
}

// Load reads graph.bin and meta.json written by Save.
func Load(dir string) (*Graph, *Meta, error) {
	f, err := os.Open(filepath.Join(dir, "graph.bin"))
	if err != nil {
		return nil, nil, fmt.Errorf("%w (run `fly build` first)", err)
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	magic := make([]byte, len(graphMagic))
	if _, err := io.ReadFull(r, magic); err != nil || string(magic) != graphMagic {
		return nil, nil, fmt.Errorf("graph.bin: bad header (rebuild with `fly build`)")
	}
	g := &Graph{}
	get := func(v any) {
		if err == nil {
			err = binary.Read(r, binary.LittleEndian, v)
		}
	}
	length := func() int {
		var n int64
		get(&n)
		return int(n)
	}
	var n int64
	get(&n)
	g.N = int(n)
	g.Start = make([]int64, length())
	get(g.Start)
	g.Target = make([]int32, length())
	get(g.Target)
	g.Weight = make([]float32, length())
	get(g.Weight)
	g.InDegree = make([]float32, length())
	get(g.InDegree)
	g.Region = make([]uint8, length())
	get(g.Region)
	for _, groups := range []*[cells][]int32{&g.OwnSensors, &g.OppSensors} {
		for c := range groups {
			groups[c] = make([]int32, length())
			get(groups[c])
		}
	}
	g.Readout = make([]int32, length())
	get(g.Readout)
	if err != nil {
		return nil, nil, fmt.Errorf("graph.bin: %w", err)
	}

	js, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		return nil, nil, err
	}
	meta := &Meta{}
	return g, meta, json.Unmarshal(js, meta)
}
