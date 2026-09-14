package readout

import (
	"math"
	"math/rand/v2"
	"testing"
)

// loss recomputes the training loss for fixed samples, with adv = ret − v held constant
// (that's what REINFORCE with a baseline differentiates).
func loss(r *Readout, samples []Sample, advs []float64, entropy float64) float64 {
	total := 0.0
	for i, s := range samples {
		p := r.Probs(s.Row)
		h := 0.0
		for _, pc := range p {
			if pc > 0 {
				h -= pc * math.Log(pc)
			}
		}
		v := r.Value(s.Row)
		total += -advs[i]*math.Log(p[s.Action]) + 0.5*(s.Return-v)*(s.Return-v) - entropy*h
	}
	return total
}

// TestGradientMatchesFiniteDifferences nudges individual weights and checks that the loss
// changes by exactly what the hand-derived gradient predicts.
func TestGradientMatchesFiniteDifferences(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 7))
	tables := GetTables()
	x, _ := FeatureMatrix("random", "")
	r := New(x)
	for c := range 9 {
		for k := range r.W[c] {
			r.W[c][k] = rng.NormFloat64() * 0.05
		}
		r.B[c] = rng.NormFloat64() * 0.1
	}
	for k := range r.U {
		r.U[k] = rng.NormFloat64() * 0.05
	}

	var samples []Sample
	var advs []float64
	for range 5 {
		row := rng.IntN(len(tables.Positions))
		legal := tables.Positions[row].Legal()
		s := Sample{Row: row, Action: legal[rng.IntN(len(legal))], Return: float64(rng.IntN(3) - 1),
			Probs: r.Probs(row), Value: r.Value(row)}
		samples = append(samples, s)
		advs = append(advs, s.Return-s.Value)
	}
	const entropy = 0.05
	g := newGrad(r.D)
	for _, s := range samples {
		r.Accumulate(g, s, entropy)
	}

	check := func(name string, param *float64, analytic float64) {
		const eps = 1e-6
		orig := *param
		*param = orig + eps
		up := loss(r, samples, advs, entropy)
		*param = orig - eps
		down := loss(r, samples, advs, entropy)
		*param = orig
		numeric := (up - down) / (2 * eps)
		if math.Abs(numeric-analytic) > 1e-5*max(1, math.Abs(numeric)) {
			t.Errorf("%s: analytic %.8f, numeric %.8f", name, analytic, numeric)
		}
	}
	for range 20 {
		c, k := rng.IntN(9), rng.IntN(r.D)
		check("W", &r.W[c][k], g.W[c][k])
		check("U", &r.U[k], g.U[k])
	}
	for c := range 9 {
		check("B", &r.B[c], g.B[c])
	}
	check("C", &r.C, g.C)
}

func TestUntrainedGreedyIsLegal(t *testing.T) {
	x, _ := FeatureMatrix("board", "")
	r := New(x)
	for row := range GetTables().Positions {
		if c := r.Greedy(row); !GetTables().Legal[row][c] {
			t.Fatalf("row %d: greedy picked illegal cell %d", row, c)
		}
	}
}
