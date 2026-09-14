package readout

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/empire/fruit-fly/internal/features"
)

// FeatureSets names the three inputs a readout can be trained on.
var FeatureSets = []string{"brain", "random", "board"}

// Labels describe the feature sets in tables.
var Labels = map[string]string{
	"brain":  "fly brain  (1,024 neurons)",
	"random": "random features (1,024, no fly)",
	"board":  "raw board  (18 bits)",
}

// FeatureMatrix returns [position][feature] inputs. cachePath is only read for "brain".
func FeatureMatrix(name, cachePath string) ([][]float64, error) {
	t := GetTables()
	switch name {
	case "board":
		return t.Board, nil
	case "random":
		// 1,024 random projections of the board, each through a ReLU. Similar boards give
		// similar features, which is exactly what lets learning generalize.
		rng := rand.New(rand.NewPCG(1234, 0))
		const d = 1024
		proj := make([][18]float64, d)
		offset := make([]float64, d)
		for k := range d {
			for b := range 18 {
				proj[k][b] = rng.NormFloat64() / math.Sqrt(18)
			}
			offset[k] = rng.NormFloat64() * 0.5
		}
		x := make([][]float64, len(t.Board))
		for i, board := range t.Board {
			x[i] = make([]float64, d)
			for k := range d {
				s := offset[k]
				for b, bit := range board {
					s += proj[k][b] * bit
				}
				x[i][k] = max(s, 0)
			}
		}
		return x, nil
	case "brain":
		c, err := features.Load(cachePath)
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

// Standardize rescales each feature to mean 0 and standard deviation 1 across positions,
// so no feature dominates just because its numbers are bigger. Constant features become 0.
func Standardize(x [][]float64) [][]float64 {
	d := len(x[0])
	mean, std := make([]float64, d), make([]float64, d)
	for _, row := range x {
		for k, v := range row {
			mean[k] += v
		}
	}
	for k := range mean {
		mean[k] /= float64(len(x))
	}
	for _, row := range x {
		for k, v := range row {
			std[k] += (v - mean[k]) * (v - mean[k])
		}
	}
	for k := range std {
		std[k] = math.Sqrt(std[k] / float64(len(x)))
		if std[k] == 0 {
			std[k] = 1
		}
	}
	out := make([][]float64, len(x))
	for i, row := range x {
		out[i] = make([]float64, d)
		for k, v := range row {
			out[i][k] = (v - mean[k]) / std[k]
		}
	}
	return out
}
