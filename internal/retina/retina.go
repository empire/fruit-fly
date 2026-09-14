// Package retina connects games to the brain: it decides which photoreceptors see which cell
// of a game's board, and turns an observation into a stimulus for the simulator.
//
// Channel c of the board uses photoreceptor pool c (R1-R6 for the mover's pieces, R8 for the
// opponent's), dealt into one equal group per cell. The dataset has no eye coordinates for
// photoreceptors, so this is a fixed arbitrary assignment, not a real spatial map of the board
// onto the retina.
package retina

import (
	"fmt"

	"github.com/empire/fruit-fly/internal/connectome"
	"github.com/empire/fruit-fly/internal/game"
	"github.com/empire/fruit-fly/internal/sim"
)

// DefaultDrive is the current into each photoreceptor under a lit cell, per channel.
var DefaultDrive = []float32{0.5, 0.5}

// Eyes maps a layout onto photoreceptor groups.
type Eyes struct {
	Layout game.Layout
	Groups [][][]int32 // [channel][cell] photoreceptors
	Drive  []float32   // [channel]
}

// New deals each channel's photoreceptor pool across the layout's cells.
func New(g *connectome.Graph, l game.Layout, drive []float32) (*Eyes, error) {
	if l.Channels > len(g.SensorPools) {
		return nil, fmt.Errorf("layout has %d channels but the eye has only %d photoreceptor pools", l.Channels, len(g.SensorPools))
	}
	if len(drive) < l.Channels {
		return nil, fmt.Errorf("%d drive values for %d channels", len(drive), l.Channels)
	}
	e := &Eyes{Layout: l, Drive: drive}
	for ch := range l.Channels {
		groups := deal(g.SensorPools[ch], l.Cells())
		if len(groups[0]) == 0 {
			return nil, fmt.Errorf("pool %d has %d photoreceptors, fewer than %d cells", ch, len(g.SensorPools[ch]), l.Cells())
		}
		e.Groups = append(e.Groups, groups)
	}
	return e, nil
}

// deal splits neurons into equal groups like dealing cards (the remainder is dropped).
func deal(neurons []int32, groups int) [][]int32 {
	out := make([][]int32, groups)
	per := len(neurons) / groups
	for k, i := range neurons[:per*groups] {
		out[k%groups] = append(out[k%groups], i)
	}
	return out
}

// Stimulus lights the photoreceptor groups of every lit cell, cell by cell.
func (e *Eyes) Stimulus(obs []bool) []sim.Input {
	cells := e.Layout.Cells()
	var inputs []sim.Input
	for cell := range cells {
		for ch := range e.Layout.Channels {
			if obs[ch*cells+cell] {
				inputs = append(inputs, sim.Input{Neurons: e.Groups[ch][cell], Current: e.Drive[ch]})
			}
		}
	}
	return inputs
}
