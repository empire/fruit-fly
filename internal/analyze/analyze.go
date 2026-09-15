// Package analyze reruns the experiments that explain the results: is the brain's output a
// smooth function of the board, and can a linear readout use it on unseen positions?
// They work on any game.Tree.
package analyze

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"

	"github.com/empire/fruit-fly/internal/featureset"
	"github.com/empire/fruit-fly/internal/game"
	"github.com/empire/fruit-fly/internal/readout"
)

// Smoothness compares how far the output moves when one opponent piece is added (an empty
// cell lit on the last channel) with how far it moves between two unrelated positions.
// Near 1.0 means the features behave like a chaotic hash (any change scrambles everything);
// well below 1.0 means similar boards give similar activity, which is what learning needs.
// It returns NaN if the game has no such neighbouring positions.
func Smoothness(t *game.Tree, x [][]float64, pairs int, seed uint64) float64 {
	rng := rand.New(rand.NewPCG(seed, 3))
	cells, last := t.Layout.Cells(), t.Layout.Channels-1
	near, far := 0.0, 0.0
	found := 0
	for tries := 0; found < pairs && tries < 1000*pairs; tries++ {
		row := rng.IntN(t.Decisions)
		obs := t.Obs[row]
		var empty []int
		for cell := range cells {
			lit := false
			for ch := range t.Layout.Channels {
				lit = lit || obs[ch*cells+cell]
			}
			if !lit {
				empty = append(empty, cell)
			}
		}
		if len(empty) < 2 {
			continue
		}
		plusOne := slices.Clone(obs)
		plusOne[last*cells+empty[rng.IntN(len(empty))]] = true
		neighbour, ok := t.RowOfObservation(plusOne)
		if !ok { // adding the piece ended the game or made an unreachable board
			continue
		}
		other := rng.IntN(t.Decisions)
		near += l1(x[row], x[neighbour])
		far += l1(x[row], x[other])
		found++
	}
	if found == 0 {
		return math.NaN()
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

func matrix(a, d int) [][]float64 {
	m := make([][]float64, a)
	for c := range m {
		m[c] = make([]float64, d)
	}
	return m
}

// Probe fits a linear model to predict which actions are optimal (multi-label logistic
// regression) on the train rows, then reports how often its top legal action is optimal on
// the test rows.
func Probe(t *game.Tree, x [][]float64, train, test []int, epochs int) float64 {
	x = readout.Standardize(x)
	d, actions := len(x[0]), t.NumActions
	w, b := matrix(actions, d), make([]float64, actions)
	m, v := matrix(actions, d), matrix(actions, d) // Adam state for w
	mb, vb := make([]float64, actions), make([]float64, actions)
	const lr, l2 = 0.01, 1e-3
	workers := runtime.GOMAXPROCS(0)
	type grad struct {
		w [][]float64
		b []float64
	}
	grads := make([]grad, workers)
	for i := range grads {
		grads[i] = grad{matrix(actions, d), make([]float64, actions)}
	}
	for epoch := 1; epoch <= epochs; epoch++ {
		// Each worker computes the gradient over its share of rows; then we sum.
		var wg sync.WaitGroup
		for i := range workers {
			wg.Go(func() {
				g := &grads[i]
				for c := range actions {
					clear(g.w[c])
					g.b[c] = 0
				}
				for j := i; j < len(train); j += workers {
					row := train[j]
					for c := range actions {
						z := b[c]
						for k, xv := range x[row] {
							z += w[c][k] * xv
						}
						y := 0.0
						if t.Best[row][c] {
							y = 1
						}
						diff := (1/(1+math.Exp(-z)) - y) / float64(len(train)*actions) // d(mean BCE)/dz
						for k, xv := range x[row] {
							g.w[c][k] += diff * xv
						}
						g.b[c] += diff
					}
				}
			})
		}
		wg.Wait()
		gw, gb := make([][]float64, actions), make([]float64, actions)
		for c := range actions {
			gw[c] = make([]float64, d)
			for i := range grads {
				for k, v := range grads[i].w[c] {
					gw[c][k] += v
				}
				gb[c] += grads[i].b[c]
			}
		}
		c1, c2 := 1-math.Pow(0.9, float64(epoch)), 1-math.Pow(0.999, float64(epoch))
		for c := range actions {
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
		for c := range actions {
			if t.Next[row][c] < 0 {
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
func Chance(t *game.Tree, rows []int) float64 {
	s := 0.0
	for _, row := range rows {
		best, legal := 0, 0
		for c := range t.NumActions {
			if t.Next[row][c] >= 0 {
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

// Run prints all three experiments for the given feature sets (keyed by featureset.Names).
func Run(t *game.Tree, sets map[string][][]float64) {
	fmt.Printf("1. smoothness: output change from one extra piece / change to an unrelated position\n")
	fmt.Printf("   (1.0 = chaotic hash, well below 1 = similar boards look similar)\n")
	for _, name := range featureset.Names {
		if v := Smoothness(t, sets[name], 300, 1); math.IsNaN(v) {
			fmt.Printf("   %-8s n/a (no reachable position differs by just one extra opponent piece)\n", name)
		} else {
			fmt.Printf("   %-8s %.2f\n", name, v)
		}
	}

	if t.Partial {
		fmt.Printf("\n2–3. held-out probe and memorization ceiling: n/a (sample tree, no perfect-play labels)\n")
		return
	}

	rng := rand.New(rand.NewPCG(0, 4))
	perm := rng.Perm(t.Decisions)
	train, test := perm[:len(perm)*4/5], perm[len(perm)*4/5:]
	fmt.Printf("\n2. held-out probe: fit on %d positions, pick an optimal move on %d unseen ones\n", len(train), len(test))
	for _, name := range featureset.Names {
		fmt.Printf("   %-8s %.1f%%\n", name, 100*Probe(t, sets[name], train, test, 300))
	}
	fmt.Printf("   chance   %.1f%%\n", 100*Chance(t, test))

	all := make([]int, t.Decisions)
	for i := range all {
		all[i] = i
	}
	fmt.Printf("\n3. ceiling: fit on all %d positions and test on the same ones (memorization)\n", t.Decisions)
	for _, name := range featureset.Names {
		fmt.Printf("   %-8s %.1f%%\n", name, 100*Probe(t, sets[name], all, all, 1000))
	}
	fmt.Printf("   chance   %.1f%%\n", 100*Chance(t, all))
}
