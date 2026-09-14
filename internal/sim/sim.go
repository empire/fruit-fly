// Package sim runs leaky integrate-and-fire (LIF) neurons over the frozen connectome.
//
// Every neuron has a membrane voltage v, like water in a leaky bucket. Each tick:
//
//	current  = sum of weights from neurons that spiked last tick
//	v        = decay*v + current + bias + sensory drive
//	spike    = v > threshold          // threshold rises if the neuron fires too often
//	v        = 0 if it spiked         // reset
//
// The simulation is event-driven: only neurons that just spiked (a few percent) walk
// their outgoing connections, so a tick costs far less than touching all 25.6M synapses.
//
// Each stimulus is simulated independently on its own goroutine. Every run starts from
// the same seeded resting state, so the stimulus is the only thing that differs, and the
// same stimulus always produces exactly the same spikes (live or cached). The simulator
// knows nothing about games: package retina turns a board into a stimulus.
//
// The parameters are engineering choices, not measured biology. They keep activity alive
// but bounded, which is all this demo needs.
package sim

import (
	"math"
	"math/rand/v2"
	"runtime"
	"sync"

	"github.com/empire/fruit-fly/internal/connectome"
)

// Params controls the neuron model.
type Params struct {
	Ticks      int
	Decay      float32 // fraction of voltage kept each tick
	Gain       float32 // base threshold = Gain * sqrt(in-degree + 1)
	BiasStd    float32 // per-neuron resting input, fixed during a run
	HomeoK     float32 // how strongly the threshold rises with the firing rate
	TargetRate float32 // firing rate (spikes per tick) the threshold steers toward
	Seed       uint64  // the fly's resting state; the same for every run
}

// DefaultParams are the settings used for the cached features.
func DefaultParams() Params {
	return Params{Ticks: 48, Decay: 0.85, Gain: 0.15, BiasStd: 0.01, HomeoK: 3, TargetRate: 0.03, Seed: 0}
}

// Input is constant extra current into a group of neurons for the whole run.
type Input struct {
	Neurons []int32
	Current float32
}

// Result is what one stimulus did to the brain.
type Result struct {
	ReadoutCounts []uint8    // spikes per readout neuron over the run
	RegionSpikes  [3][]int32 // spikes per region (optic/central/vnc) per tick
}

// Brain is the frozen connectome plus neuron parameters. It is safe for concurrent use.
type Brain struct {
	G     *connectome.Graph
	P     Params
	theta []float32 // base threshold per neuron
	bias  []float32 // resting input per neuron (same for every run)
	v0    []float32 // starting voltage per neuron (same for every run)
}

// New prepares a brain. Params can't change afterwards.
func New(g *connectome.Graph, p Params) *Brain {
	b := &Brain{G: g, P: p, theta: make([]float32, g.N), bias: make([]float32, g.N), v0: make([]float32, g.N)}
	rng := rand.New(rand.NewPCG(p.Seed, 0x5eed))
	for i := range g.N {
		b.theta[i] = p.Gain * float32(math.Sqrt(float64(g.InDegree[i])+1))
		b.bias[i] = float32(rng.NormFloat64()) * p.BiasStd
		b.v0[i] = rng.Float32() * 0.05
	}
	return b
}

// state holds one run's working buffers, reused across runs.
type state struct {
	v, rate, input, current []float32
	spikes                  []int32
	fired                   []bool
}

func (b *Brain) newState() *state {
	n := b.G.N
	return &state{v: make([]float32, n), rate: make([]float32, n), input: make([]float32, n),
		current: make([]float32, n), spikes: make([]int32, 0, n/10), fired: make([]bool, n)}
}

// Run simulates one stimulus.
func (b *Brain) Run(stimulus []Input) Result { return b.run(b.newState(), stimulus) }

func (b *Brain) run(s *state, stimulus []Input) Result {
	g, p := b.G, b.P
	copy(s.v, b.v0)
	copy(s.input, b.bias)
	clear(s.current)
	for i := range s.rate {
		s.rate[i] = p.TargetRate
	}
	s.spikes = s.spikes[:0]

	// The stimulus: constant extra input, e.g. photoreceptors under pieces on the board.
	for _, in := range stimulus {
		for _, i := range in.Neurons {
			s.input[i] += in.Current
		}
	}

	res := Result{ReadoutCounts: make([]uint8, len(g.Readout))}
	for r := range res.RegionSpikes {
		res.RegionSpikes[r] = make([]int32, p.Ticks)
	}
	fired := s.fired
	for t := range p.Ticks {
		// 1. Deliver last tick's spikes along outgoing synapses.
		for _, i := range s.spikes {
			for k := g.Start[i]; k < g.Start[i+1]; k++ {
				s.current[g.Target[k]] += g.Weight[k]
			}
		}
		// 2. Integrate, compare with the adaptive threshold, fire and reset.
		s.spikes = s.spikes[:0]
		for i := range g.N {
			v := p.Decay*s.v[i] + s.current[i] + s.input[i]
			s.current[i] = 0
			threshold := b.theta[i] * float32(math.Exp(float64(p.HomeoK*(s.rate[i]-p.TargetRate))))
			spiked := v > threshold
			fired[i] = spiked
			s.rate[i] *= 0.97
			if spiked {
				v = 0
				s.rate[i] += 0.03
				s.spikes = append(s.spikes, int32(i))
				res.RegionSpikes[g.Region[i]][t]++
			}
			s.v[i] = v
		}
		// 3. Count what the readout neurons did.
		for k, i := range g.Readout {
			if fired[i] {
				res.ReadoutCounts[k]++
			}
		}
	}
	return res
}

// RunMany simulates stimuli in parallel, one goroutine per CPU. Results keep input order.
// progress, if not nil, is called after each finished run.
func (b *Brain) RunMany(stimuli [][]Input, progress func()) []Result {
	results := make([]Result, len(stimuli))
	jobs := make(chan int)
	var wg sync.WaitGroup
	for range runtime.GOMAXPROCS(0) {
		wg.Go(func() {
			s := b.newState()
			for j := range jobs {
				results[j] = b.run(s, stimuli[j])
				if progress != nil {
					progress()
				}
			}
		})
	}
	for j := range stimuli {
		jobs <- j
	}
	close(jobs)
	wg.Wait()
	return results
}
