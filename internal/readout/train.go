package readout

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"sync"
	"time"

	"github.com/empire/fruit-fly/internal/featureset"
	"github.com/empire/fruit-fly/internal/game"
)

// Opponent is who the learner plays against in a game.
type Opponent int

const (
	Self Opponent = iota // the learner plays both sides
	Random
	Perfect
)

// Chooser picks an action at the current position of a Walk.
type Chooser func(w *game.Walk, rng *rand.Rand) int

func opponentMove(t *game.Tree, o Opponent, w *game.Walk, rng *rand.Rand) int {
	if w.Row < 0 {
		return w.Legal[rng.IntN(len(w.Legal))]
	}
	if o == Perfect {
		return game.PerfectMove(t, w.Row, rng)
	}
	return game.RandomMove(t, w.Row, rng)
}

// playGame plays one game on t and returns the result for the first player (+1, 0, -1).
// If learn is set, it records every learner move of r as a Sample with its Return filled in.
func playGame(t *game.Tree, fly Chooser, r *Readout, o Opponent, flyFirst bool, rng *rand.Rand, learn bool) (int, []Sample) {
	var (
		w       = t.NewWalk()
		samples []Sample
		steps   []int
	)
	for step := 0; ; step++ {
		if w.Over {
			outcomeFirst := w.Value
			if step%2 == 1 {
				outcomeFirst = -w.Value
			}
			for i := range samples {
				samples[i].Return = float64(outcomeFirst)
				if steps[i]%2 == 1 {
					samples[i].Return = -samples[i].Return
				}
			}
			return outcomeFirst, samples
		}
		firstMoves := step%2 == 0
		var action int
		if o == Self || flyFirst == firstMoves {
			if learn && w.Row < 0 && r.enc == nil {
				action = w.Legal[rng.IntN(len(w.Legal))]
			} else if learn {
				p := r.ProbsWalk(w)
				action = sampleAction(p, rng)
				s := Sample{Row: w.Row, Action: action, Probs: p, Value: r.ValueWalk(w)}
				if w.Row < 0 {
					s.X = r.scaled(w.Obs())
					s.Row = 0
				}
				samples = append(samples, s)
				steps = append(steps, step)
			} else {
				action = fly(w, rng)
			}
		} else {
			action = opponentMove(t, o, w, rng)
		}
		w.Step(action)
	}
}

func sampleAction(p []float64, rng *rand.Rand) int {
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
	return last
}

// TrainConfig controls REINFORCE training.
type TrainConfig struct {
	Games    int
	Batch    int
	LR       float64
	Entropy  float64
	Seed     uint64
	LogEvery int // batches between win/loss snapshots
}

// DefaultTrainConfig trains for 300,000 games in batches of 512.
func DefaultTrainConfig() TrainConfig {
	return TrainConfig{Games: 300_000, Batch: 512, LR: 0.01, Entropy: 0.01, LogEvery: 50}
}

// Train learns a readout for t with self-play REINFORCE, rotating opponents between the
// learner itself (teaches both sides), a random player and a perfect player (keep it honest).
func Train(name string, t *game.Tree, x [][]float64, cfg TrainConfig) *Readout {
	r := New(t, x)
	if enc, err := featureset.Encoder(name, t.Layout); err == nil && enc != nil {
		r.AttachEncoder(enc)
	}
	opt := newAdam(r.A, r.D, cfg.LR)
	workers := runtime.GOMAXPROCS(0)
	grads := make([]*Grad, workers)
	for w := range grads {
		grads[w] = newGrad(r.A, r.D)
	}
	seedRng := rand.New(rand.NewPCG(cfg.Seed, 1))

	start := time.Now()
	trained := 0
	progress := func() {
		elapsed := time.Since(start).Seconds()
		left := ""
		if trained > 0 && elapsed > 0 {
			sec := elapsed * float64(cfg.Games-trained) / float64(trained)
			if sec >= 1 {
				left = fmt.Sprintf("  ~%.0fs left", sec)
			}
		}
		fmt.Printf("\r  trained %8d / %d games%s\033[K", trained, cfg.Games, left)
	}

	for it := range cfg.Games / cfg.Batch {
		opponent := Opponent(it % 3)
		counts := make([]int, workers)
		var wg sync.WaitGroup
		for w := range workers {
			seed := seedRng.Uint64()
			wg.Go(func() {
				rng := rand.New(rand.NewPCG(seed, uint64(w)))
				g := grads[w]
				g.zero()
				for range cfg.Batch / workers {
					_, samples := playGame(t, nil, r, opponent, rng.IntN(2) == 0, rng, true)
					for _, s := range samples {
						r.Accumulate(g, s, cfg.Entropy)
					}
					counts[w] += len(samples)
				}
			})
		}
		wg.Wait()

		total := newGrad(r.A, r.D)
		n := 0
		for w, g := range grads {
			n += counts[w]
			for c := range r.A {
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

		trained += cfg.Batch
		progress()
		if cfg.LogEvery > 0 && (it+1)%cfg.LogEvery == 0 {
			s := Evaluate(t, r.Chooser(), 400, uint64(it))
			fmt.Printf("\r[%s] games %8d / %d  vs random: win %3.0f%%  vs perfect: loss %3.0f%%\033[K\n",
				name, trained, cfg.Games, 100*s.VsRandom.Win, 100*s.VsPerfect.Loss)
		}
	}
	fmt.Printf("\r[%s] trained %d games in %.1fs\033[K\n", name, trained, time.Since(start).Seconds())
	return r
}

// Chooser plays the readout's most likely move, including off-tree positions.
func (r *Readout) Chooser() Chooser {
	return func(w *game.Walk, _ *rand.Rand) int { return r.GreedyWalk(w) }
}
