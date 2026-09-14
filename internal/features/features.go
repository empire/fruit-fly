// Package features simulates the brain once for every position and caches what the
// readout neurons did.
//
// Tic-tac-toe has 4,520 positions where someone must move, and the wiring never changes,
// so every question we will ever ask the brain can be answered in advance. Training then
// only needs table lookups.
package features

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"sync/atomic"
	"time"

	"github.com/empire/fruit-fly/internal/game"
	"github.com/empire/fruit-fly/internal/sim"
)

const magic = "FLYFEAT1"

// Cache holds the brain's response to every decision position, in game.Positions() order.
type Cache struct {
	Ticks   int
	Counts  [][]uint8    // [position][readout neuron] spike counts
	Regions [][3][]int32 // [position][region][tick] spikes
}

// Compute simulates every position and writes the cache to path.
func Compute(brain *sim.Brain, path string) (*Cache, error) {
	positions := game.Positions()
	var done atomic.Int64
	start := time.Now()
	stop := make(chan struct{})
	go func() {
		tick := time.NewTicker(time.Second)
		defer tick.Stop()
		for {
			select {
			case <-stop:
				return
			case <-tick.C:
				n := done.Load()
				fmt.Printf("\r  simulated %d / %d positions (%.0fs)", n, len(positions), time.Since(start).Seconds())
			}
		}
	}()
	results := brain.RunMany(positions, func() { done.Add(1) })
	close(stop)
	fmt.Printf("\r  simulated %d positions x %d ticks in %.1fs\n", len(positions), brain.P.Ticks, time.Since(start).Seconds())

	c := &Cache{Ticks: brain.P.Ticks}
	for _, r := range results {
		c.Counts = append(c.Counts, r.ReadoutCounts)
		c.Regions = append(c.Regions, r.RegionSpikes)
	}
	return c, c.Save(path)
}

// Save writes the cache as little-endian binary.
func (c *Cache) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	put := func(v any) {
		if err == nil {
			err = binary.Write(w, binary.LittleEndian, v)
		}
	}
	put([]byte(magic))
	put([3]int64{int64(len(c.Counts)), int64(len(c.Counts[0])), int64(c.Ticks)})
	for i := range c.Counts {
		put(c.Counts[i])
		for r := range 3 {
			put(c.Regions[i][r])
		}
	}
	if err == nil {
		err = w.Flush()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}

// Load reads a cache written by Save.
func Load(path string) (*Cache, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("%w (run `fly features` first)", err)
	}
	defer f.Close()
	r := bufio.NewReader(f)
	head := make([]byte, len(magic))
	if _, err := io.ReadFull(r, head); err != nil || string(head) != magic {
		return nil, fmt.Errorf("%s: bad header", path)
	}
	var dims [3]int64
	get := func(v any) {
		if err == nil {
			err = binary.Read(r, binary.LittleEndian, v)
		}
	}
	get(&dims)
	if err != nil {
		return nil, err
	}
	c := &Cache{Ticks: int(dims[2])}
	c.Counts = make([][]uint8, dims[0])
	c.Regions = make([][3][]int32, dims[0])
	for i := range c.Counts {
		c.Counts[i] = make([]uint8, dims[1])
		get(c.Counts[i])
		for reg := range 3 {
			c.Regions[i][reg] = make([]int32, c.Ticks)
			get(c.Regions[i][reg])
		}
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(c.Counts) != len(game.Positions()) {
		return nil, fmt.Errorf("%s: %d positions, want %d", path, len(c.Counts), len(game.Positions()))
	}
	return c, nil
}

// Sanity checks that the brain's output actually depends on the board.
// A readout trained on features that don't change is meaningless.
func Sanity(c *Cache) bool {
	positions, neurons := len(c.Counts), len(c.Counts[0])
	active, varying := 0, 0
	for k := range neurons {
		lo, hi, sum := uint8(255), uint8(0), 0
		for i := range positions {
			v := c.Counts[i][k]
			lo, hi, sum = min(lo, v), max(hi, v), sum+int(v)
		}
		if sum > 0 {
			active++
		}
		if hi > lo {
			varying++
		}
	}
	distinct := map[string]bool{}
	total := 0
	for _, row := range c.Counts {
		distinct[string(row)] = true
		for _, v := range row {
			total += int(v)
		}
	}
	rng := rand.New(rand.NewPCG(0, 0))
	dists := make([]int, 2000)
	for d := range dists {
		a, b := c.Counts[rng.IntN(positions)], c.Counts[rng.IntN(positions)]
		for k := range a {
			dists[d] += int(math.Abs(float64(a[k]) - float64(b[k])))
		}
	}
	slices.Sort(dists)

	fmt.Printf("readout neurons that ever fire:       %.1f%%\n", 100*float64(active)/float64(neurons))
	fmt.Printf("readout neurons that vary by board:   %.1f%%\n", 100*float64(varying)/float64(neurons))
	fmt.Printf("distinct activity patterns:           %d / %d positions\n", len(distinct), positions)
	fmt.Printf("mean spikes per readout neuron:       %.2f\n", float64(total)/float64(positions*neurons))
	fmt.Printf("L1 distance between random positions: median %d\n", dists[len(dists)/2])
	ok := float64(varying)/float64(neurons) > 0.05 && float64(len(distinct)) > 0.9*float64(positions)
	if ok {
		fmt.Println("sanity: PASS")
	} else {
		fmt.Println("sanity: FAIL (brain output barely depends on the board)")
	}
	return ok
}

// Bench times single-board runs and the parallel full precompute estimate.
func Bench(brain *sim.Brain) {
	positions := game.Positions()
	brain.Run(positions[100]) // warm up
	start := time.Now()
	const n = 4
	for i := range n {
		brain.Run(positions[i*1000])
	}
	one := time.Since(start) / n
	start = time.Now()
	brain.RunMany(positions[:64], nil)
	many := time.Since(start) / 64
	fmt.Printf("one board, one goroutine:  %v (%v per tick)\n", one.Round(time.Millisecond), (one / time.Duration(brain.P.Ticks)).Round(10*time.Microsecond))
	fmt.Printf("parallel, per board:       %v\n", many.Round(time.Millisecond))
	fmt.Printf("all %d positions:        ~%.0fs\n", len(positions), (many * time.Duration(len(positions))).Seconds())
}
