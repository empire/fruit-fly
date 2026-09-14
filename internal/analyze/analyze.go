// Package analyze reruns the experiments that explain the results: is the brain's output a
// smooth function of the board, and can a linear readout use it on unseen positions?
package analyze

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"

	"github.com/empire/fruit-fly/internal/game"
	"github.com/empire/fruit-fly/internal/readout"
)

// Smoothness compares how far the brain's output moves when one opponent piece is added
// with how far it moves between two unrelated positions. Near 1.0 means the brain behaves
// like a chaotic hash (any change scrambles everything); well below 1.0 means similar
// boards give similar activity, which is what learning needs.
func Smoothness(brainX [][]float64, pairs int, seed uint64) float64 {
	t := readout.GetTables()
	rng := rand.New(rand.NewPCG(seed, 3))
	near, far := 0.0, 0.0
	for found := 0; found < pairs; {
		p := t.Positions[rng.IntN(len(t.Positions))]
		legal := p.Legal()
		if len(legal) < 2 {
			continue
		}
		plusOne := game.Pos{Mine: p.Mine, Opp: p.Opp | 1<<legal[rng.IntN(len(legal))]}
		neighbour := t.RowOf(plusOne)
		if neighbour < 0 { // adding the piece ended the game or made an unreachable board
			continue
		}
		other := rng.IntN(len(t.Positions))
		near += l1(brainX[t.RowOf(p)], brainX[neighbour])
		far += l1(brainX[t.RowOf(p)], brainX[other])
		found++
	}
	return near / far
}

func l1(a, b []float64) float64 {
	s := 0.0
	for k := range a {
		s += math.Abs(a[k] - b[k])
	}
	return s
}

// Probe fits a linear model to predict which cells are optimal moves (multi-label logistic
// regression) on the train rows, then reports how often its top legal cell is optimal on
// the test rows.
func Probe(x [][]float64, train, test []int, epochs int) float64 {
	t := readout.GetTables()
	x = readout.Standardize(x)
	d := len(x[0])
	var w [9][]float64
	var b [9]float64
	for c := range w {
		w[c] = make([]float64, d)
	}
	var m, v [9][]float64 // Adam state for w
	var mb, vb [9]float64
	for c := range m {
		m[c], v[c] = make([]float64, d), make([]float64, d)
	}
	const lr, l2 = 0.01, 1e-3
	workers := runtime.GOMAXPROCS(0)
	type grad struct {
		w [9][]float64
		b [9]float64
	}
	grads := make([]grad, workers)
	for i := range grads {
		for c := range 9 {
			grads[i].w[c] = make([]float64, d)
		}
	}
	for epoch := 1; epoch <= epochs; epoch++ {
		// Each worker computes the gradient over its share of rows; then we sum.
		var wg sync.WaitGroup
		for i := range workers {
			wg.Go(func() {
				g := &grads[i]
				for c := range 9 {
					clear(g.w[c])
					g.b[c] = 0
				}
				for j := i; j < len(train); j += workers {
					row := train[j]
					for c := range 9 {
						z := b[c]
						for k, xv := range x[row] {
							z += w[c][k] * xv
						}
						y := 0.0
						if t.Best[row][c] {
							y = 1
						}
						diff := (1/(1+math.Exp(-z)) - y) / float64(len(train)*9) // d(mean BCE)/dz
						for k, xv := range x[row] {
							g.w[c][k] += diff * xv
						}
						g.b[c] += diff
					}
				}
			})
		}
		wg.Wait()
		var gw [9][]float64
		var gb [9]float64
		for c := range 9 {
			gw[c] = make([]float64, d)
			for i := range grads {
				for k, v := range grads[i].w[c] {
					gw[c][k] += v
				}
				gb[c] += grads[i].b[c]
			}
		}
		c1, c2 := 1-math.Pow(0.9, float64(epoch)), 1-math.Pow(0.999, float64(epoch))
		for c := range 9 {
			for k := range d {
				g := gw[c][k] + 2*l2*w[c][k]
				m[c][k] = 0.9*m[c][k] + 0.1*g
				v[c][k] = 0.999*v[c][k] + 0.001*g*g
				w[c][k] -= lr * (m[c][k] / c1) / (math.Sqrt(v[c][k]/c2) + 1e-8)
			}
			mb[c] = 0.9*mb[c] + 0.1*gb[c]
			vb[c] = 0.999*vb[c] + 0.001*gb[c]*gb[c]
			b[c] -= lr * (mb[c] / c1) / (math.Sqrt(vb[c]/c2) + 1e-8)
		}
	}

	hits := 0
	for _, row := range test {
		best, bestZ := -1, math.Inf(-1)
		for c := range 9 {
			if !t.Legal[row][c] {
				continue
			}
			z := b[c]
			for k, xv := range x[row] {
				z += w[c][k] * xv
			}
			if z > bestZ {
				best, bestZ = c, z
			}
		}
		if t.Best[row][best] {
			hits++
		}
	}
	return float64(hits) / float64(len(test))
}

// Chance is how often a uniformly random legal move is optimal, averaged over rows.
func Chance(rows []int) float64 {
	t := readout.GetTables()
	s := 0.0
	for _, row := range rows {
		best, legal := 0, 0
		for c := range 9 {
			if t.Legal[row][c] {
				legal++
				if t.Best[row][c] {
					best++
				}
			}
		}
		s += float64(best) / float64(legal)
	}
	return s / float64(len(rows))
}

// Run prints all three experiments for the given feature sets.
func Run(sets map[string][][]float64) {
	t := readout.GetTables()
	fmt.Printf("1. smoothness: output change from one extra piece / change to an unrelated position\n")
	fmt.Printf("   (1.0 = chaotic hash, well below 1 = similar boards look similar)\n")
	for _, name := range readout.FeatureSets {
		fmt.Printf("   %-8s %.2f\n", name, Smoothness(sets[name], 300, 1))
	}

	rng := rand.New(rand.NewPCG(0, 4))
	perm := rng.Perm(len(t.Positions))
	train, test := perm[:3600], perm[3600:]
	fmt.Printf("\n2. held-out probe: fit on 3,600 positions, pick an optimal move on 920 unseen ones\n")
	for _, name := range readout.FeatureSets {
		fmt.Printf("   %-8s %.1f%%\n", name, 100*Probe(sets[name], train, test, 300))
	}
	fmt.Printf("   chance   %.1f%%\n", 100*Chance(test))

	all := make([]int, len(t.Positions))
	for i := range all {
		all[i] = i
	}
	fmt.Printf("\n3. ceiling: fit on all 4,520 positions and test on the same ones (memorization)\n")
	for _, name := range readout.FeatureSets {
		fmt.Printf("   %-8s %.1f%%\n", name, 100*Probe(sets[name], all, all, 1000))
	}
	fmt.Printf("   chance   %.1f%%\n", 100*Chance(all))
}
