// Package readout is the only part that learns: a linear map from neuron activity to a cell.
//
//	scores = W · standardize(features[position]) + b     // 9 scores, one per cell
//	illegal cells -> -Inf, softmax -> probabilities, sample (training) or argmax (playing)
//
// Training is REINFORCE: play a batch of games, then make each move the learner made more
// likely in proportion to (final result − predicted result). The prediction comes from a
// second linear head (the value baseline), which reduces noise. Gradients are written by
// hand, because the model is small enough to derive them on paper (see policy.go).
//
// The same code trains three readouts, so we can tell whether the fly brain helps:
//
//	brain   1,024 fly neuron spike counts                (the experiment)
//	random  1,024 random ReLU features of the raw board   (same size, no fly)
//	board   the raw 18 board bits                          (smallest possible)
package readout

import (
	"sync"

	"github.com/empire/fruit-fly/internal/game"
)

// Tables are lookup tables over the 4,520 decision positions, in game.Positions() order.
type Tables struct {
	Positions []game.Pos
	Row       []int32     // [2^18] position key -> row, -1 if not a decision position
	Legal     [][9]bool   // [P] empty cells
	Best      [][9]bool   // [P] optimal moves under perfect play
	Board     [][]float64 // [P][18] my bits then opponent bits
}

// RowOf returns the table row of a decision position.
func (t *Tables) RowOf(p game.Pos) int { return int(t.Row[p.Key()]) }

// GetTables builds the tables once.
var GetTables = sync.OnceValue(func() *Tables {
	positions := game.Positions()
	t := &Tables{Positions: positions, Row: make([]int32, 1<<18),
		Legal: make([][9]bool, len(positions)), Best: make([][9]bool, len(positions)),
		Board: make([][]float64, len(positions))}
	for i := range t.Row {
		t.Row[i] = -1
	}
	for i, p := range positions {
		t.Row[p.Key()] = int32(i)
		for _, c := range p.Legal() {
			t.Legal[i][c] = true
		}
		for _, c := range game.BestMoves(p) {
			t.Best[i][c] = true
		}
		t.Board[i] = make([]float64, 18)
		for b := range 18 {
			t.Board[i][b] = float64(p.Key() >> b & 1)
		}
	}
	return t
})
