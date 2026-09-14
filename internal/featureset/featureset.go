// Package featureset builds the inputs a readout can be trained on, for any game.
//
//	brain   1,024 fly neuron spike counts                   (the experiment)
//	random  1,024 random ReLU features of the observation    (same size, no fly)
//	board   the raw observation bits                          (smallest possible)
//
// The two controls matter: without them, "the fly wins" can't be told apart from "any
// fixed features of this size win".
package featureset

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/empire/fruit-fly/internal/features"
	"github.com/empire/fruit-fly/internal/game"
)

// Names lists the feature sets.
var Names = []string{"brain", "random", "board"}

// randomDim is the number of random features, matched to the 1,024 readout neurons.
const randomDim = 1024

// Label describes a feature set in tables.
func Label(name string, t *game.Tree) string {
	switch name {
	case "brain":
		return "fly brain  (1,024 neurons)"
	case "random":
		return "random features (1,024, no fly)"
	case "board":
		return fmt.Sprintf("raw board  (%d bits)", t.Layout.Size())
	}
	return name
}

// Board returns the observation bits of every row as numbers.
func Board(t *game.Tree) [][]float64 {
	x := make([][]float64, t.Decisions)
	for i, obs := range t.Obs {
		x[i] = make([]float64, len(obs))
		for b, lit := range obs {
			if lit {
				x[i][b] = 1
			}
		}
	}
	return x
}

// Matrix returns [row][feature] inputs for t. cachePath is only read for "brain".
func Matrix(name string, t *game.Tree, cachePath string) ([][]float64, error) {
	switch name {
	case "board":
		return Board(t), nil
	case "random":
		// Random projections of the board, each through a ReLU. Similar boards give
		// similar features, which is exactly what lets learning generalize.
		board := Board(t)
		bits := t.Layout.Size()
		rng := rand.New(rand.NewPCG(1234, 0))
		proj := make([][]float64, randomDim)
		offset := make([]float64, randomDim)
		for k := range randomDim {
			proj[k] = make([]float64, bits)
			for b := range bits {
				proj[k][b] = rng.NormFloat64() / math.Sqrt(float64(bits))
			}
			offset[k] = rng.NormFloat64() * 0.5
		}
		x := make([][]float64, len(board))
		for i, row := range board {
			x[i] = make([]float64, randomDim)
			for k := range randomDim {
				s := offset[k]
				for b, bit := range row {
					s += proj[k][b] * bit
				}
				x[i][k] = max(s, 0)
			}
		}
		return x, nil
	case "brain":
		c, err := features.Load(cachePath, t)
		if err != nil {
			return nil, err
		}
		x := make([][]float64, len(c.Counts))
		for i, counts := range c.Counts {
			x[i] = make([]float64, len(counts))
			for k, v := range counts {
				x[i][k] = float64(v)
			}
		}
		return x, nil
	}
	return nil, fmt.Errorf("unknown feature set %q (want brain, random or board)", name)
}
