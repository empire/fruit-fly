package readout

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"sync"

	"github.com/empire/fruit-fly/internal/game"
)

// Opponent is who the learner plays against in a game.
type Opponent int

const (
	Self Opponent = iota // the learner plays both sides
	Random
	Perfect
)

// Chooser picks a cell for a decision position (given by table row).
type Chooser func(row int, rng *rand.Rand) int

func opponentMove(o Opponent, row int, rng *rand.Rand) int {
	t := GetTables()
	allowed := t.Legal[row]
	if o == Perfect {
		allowed = t.Best[row]
	}
	var cells []int
	for c, ok := range allowed {
		if ok {
			cells = append(cells, c)
		}
	}
	return cells[rng.IntN(len(cells))]
}

// playGame plays one game and returns the result for X (+1, 0, -1).
// If learn is set, it records every learner move as a Sample with its Return filled in.
func playGame(fly Chooser, r *Readout, o Opponent, flyIsX bool, rng *rand.Rand, learn bool) (int, []Sample) {
	t := GetTables()
	var (
		pos     game.Pos
		samples []Sample
		steps   []int
	)
	for step := 0; ; step++ {
		if v, over := pos.Terminal(); over {
			outcomeX := v // v is for the side to move, which just lost or drew
			if step%2 == 1 {
				outcomeX = -v
			}
			for i := range samples {
				samples[i].Return = float64(outcomeX)
				if steps[i]%2 == 1 { // O moved: flip to O's point of view
					samples[i].Return = -samples[i].Return
				}
			}
			return outcomeX, samples
		}
		row := t.RowOf(pos)
		xMoves := step%2 == 0
		var cell int
		if o == Self || flyIsX == xMoves {
			if learn {
				p := r.Probs(row)
				cell = sampleCell(p, rng)
				samples = append(samples, Sample{Row: row, Action: cell, Probs: p, Value: r.Value(row)})
				steps = append(steps, step)
			} else {
				cell = fly(row, rng)
			}
		} else {
			cell = opponentMove(o, row, rng)
		}
		pos = pos.Play(cell)
	}
}

func sampleCell(p [9]float64, rng *rand.Rand) int {
	u := rng.Float64()
	last := 0
	for c, pc := range p {
		if pc == 0 {
			continue
		}
		last = c
		if u < pc {
			return c
		}
		u -= pc
	}
	return last // rounding
}

// TrainConfig controls REINFORCE training.
type TrainConfig struct {
	Games    int
	Batch    int
	LR       float64
	Entropy  float64
	Seed     uint64
	LogEvery int // batches between progress lines
}

// DefaultTrainConfig trains for 300,000 games in batches of 512.
func DefaultTrainConfig() TrainConfig {
	return TrainConfig{Games: 300_000, Batch: 512, LR: 0.01, Entropy: 0.01, LogEvery: 50}
}

// Train learns a readout with self-play REINFORCE, rotating opponents between the learner
// itself (teaches both sides), a random player and a perfect player (keep it honest).
func Train(name string, x [][]float64, cfg TrainConfig) *Readout {
	r := New(x)
	opt := newAdam(r.D, cfg.LR)
	workers := runtime.GOMAXPROCS(0)
	grads := make([]*Grad, workers)
	for w := range grads {
		grads[w] = newGrad(r.D)
	}
	seedRng := rand.New(rand.NewPCG(cfg.Seed, 1))

	for it := range cfg.Games / cfg.Batch {
		opponent := Opponent(it % 3)
		counts := make([]int, workers)
		var wg sync.WaitGroup
		for w := range workers {
			seed := seedRng.Uint64()
			wg.Go(func() {
				rng := rand.New(rand.NewPCG(seed, uint64(w)))
				g := grads[w]
				*g = Grad{W: g.W, U: g.U} // zero the scalars, keep the slices
				for c := range 9 {
					clear(g.W[c])
				}
				clear(g.U)
				for range cfg.Batch / workers {
					_, samples := playGame(nil, r, opponent, rng.IntN(2) == 0, rng, true)
					for _, s := range samples {
						r.Accumulate(g, s, cfg.Entropy)
					}
					counts[w] += len(samples)
				}
			})
		}
		wg.Wait()

		total := newGrad(r.D)
		n := 0
		for w, g := range grads {
			n += counts[w]
			for c := range 9 {
				for k, v := range g.W[c] {
					total.W[c][k] += v
				}
				total.B[c] += g.B[c]
			}
			for k, v := range g.U {
				total.U[k] += v
			}
			total.C += g.C
		}
		opt.step(r, total, 1/float64(n))

		if (it+1)%cfg.LogEvery == 0 {
			s := Evaluate(r.Chooser(), 400, uint64(it))
			fmt.Printf("[%s] games %8d  vs random: win %3.0f%%  vs perfect: loss %3.0f%%\n",
				name, (it+1)*cfg.Batch, 100*s.VsRandom.Win, 100*s.VsPerfect.Loss)
		}
	}
	return r
}

// Chooser plays the readout's most likely move.
func (r *Readout) Chooser() Chooser { return func(row int, _ *rand.Rand) int { return r.Greedy(row) } }
