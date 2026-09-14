package sim

import (
	"slices"
	"testing"

	"github.com/empire/fruit-fly/internal/connectome"
	"github.com/empire/fruit-fly/internal/game"
)

// A hand-built brain: 99 photoreceptors (54 R1-R6, 45 R8) plus three downstream neurons.
const (
	excited   = 99
	inhibitor = 100
	target    = 101
)

type synapse struct {
	from, to int32
	w        float32
}

func toyGraph(synapses []synapse) *connectome.Graph {
	const n = 102
	slices.SortStableFunc(synapses, func(a, b synapse) int { return int(a.from - b.from) })
	g := &connectome.Graph{N: n, Start: make([]int64, n+1), InDegree: make([]float32, n),
		Region: make([]uint8, n), Readout: []int32{excited, inhibitor, target}}
	for _, s := range synapses {
		g.Start[s.from+1]++
		g.Target = append(g.Target, s.to)
		g.Weight = append(g.Weight, s.w)
	}
	for i := range n {
		g.Start[i+1] += g.Start[i]
	}
	for c := range 9 {
		for k := range 6 {
			g.OwnSensors[c] = append(g.OwnSensors[c], int32(c*6+k))
		}
		for k := range 5 {
			g.OppSensors[c] = append(g.OppSensors[c], int32(54+c*5+k))
		}
	}
	return g
}

var quiet = Params{Ticks: 20, Decay: 0.85, Gain: 0.5, OwnDrive: 1, OppDrive: 1, TargetRate: 0.03}

func TestOwnPieceExcitesDownstreamOnlyWhenPresent(t *testing.T) {
	var syn []synapse
	for s := int32(24); s < 30; s++ { // the six R1-R6 cells of board cell 4
		syn = append(syn, synapse{s, excited, 0.4})
	}
	brain := New(toyGraph(syn), quiet)
	if got := brain.Run(game.Pos{Mine: 1 << 4}).ReadoutCounts[0]; got == 0 {
		t.Error("piece on cell 4: expected spikes downstream")
	}
	if got := brain.Run(game.Pos{Mine: 1 << 3}).ReadoutCounts[0]; got != 0 {
		t.Errorf("piece elsewhere: got %d spikes, want 0", got)
	}
}

func TestInhibitionCancelsExcitation(t *testing.T) {
	var syn []synapse
	for s := int32(0); s < 6; s++ { // own piece in cell 0 excites target
		syn = append(syn, synapse{s, target, 0.4})
	}
	for s := int32(54); s < 59; s++ { // opponent piece in cell 0 drives the inhibitor ...
		syn = append(syn, synapse{s, inhibitor, 0.4})
	}
	syn = append(syn, synapse{inhibitor, target, -5}) // ... which silences target
	brain := New(toyGraph(syn), quiet)
	alone := brain.Run(game.Pos{Mine: 1}).ReadoutCounts[2]
	blocked := brain.Run(game.Pos{Mine: 1, Opp: 1}).ReadoutCounts[2]
	if alone <= blocked {
		t.Errorf("target spikes: alone %d, with inhibition %d; want alone > blocked", alone, blocked)
	}
}

func TestSameBoardSameSpikesRegardlessOfBatch(t *testing.T) {
	var syn []synapse
	for s := int32(0); s < 54; s++ {
		syn = append(syn, synapse{s, excited, 0.3})
	}
	p := DefaultParams()
	p.Ticks = 30
	brain := New(toyGraph(syn), p)
	boards := []game.Pos{{Mine: 0b101, Opp: 0b010}, {Mine: 1}}
	many := brain.RunMany(boards, nil)
	for i, b := range boards {
		if !slices.Equal(brain.Run(b).ReadoutCounts, many[i].ReadoutCounts) {
			t.Errorf("board %d: Run and RunMany disagree", i)
		}
	}
}
