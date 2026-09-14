package readout

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/empire/fruit-fly/internal/featureset"
	"github.com/empire/fruit-fly/internal/game"
	"github.com/empire/fruit-fly/internal/games/hexapawn"
	"github.com/empire/fruit-fly/internal/games/tictactoe"
)

func trees(t *testing.T) []*game.Tree {
	t.Helper()
	ttt, err := game.Compile(tictactoe.New())
	if err != nil {
		t.Fatal(err)
	}
	hp, _ := hexapawn.New(3, 4)
	hex, err := game.Compile(hp)
	if err != nil {
		t.Fatal(err)
	}
	return []*game.Tree{ttt, hex}
}

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
	for _, tree := range trees(t) {
		t.Run(tree.Name, func(t *testing.T) { checkGradient(t, tree) })
	}
}

func checkGradient(t *testing.T, tree *game.Tree) {
	rng := rand.New(rand.NewPCG(7, 7))
	x, _ := featureset.Matrix("random", tree, "")
	r := New(tree, x)
	for c := range r.A {
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
		row := rng.IntN(tree.Decisions)
		legal := tree.LegalActions(row)
		s := Sample{Row: row, Action: legal[rng.IntN(len(legal))], Return: float64(rng.IntN(3) - 1),
			Probs: r.Probs(row), Value: r.Value(row)}
		samples = append(samples, s)
		advs = append(advs, s.Return-s.Value)
	}
	const entropy = 0.05
	g := newGrad(r.A, r.D)
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
		c, k := rng.IntN(r.A), rng.IntN(r.D)
		check("W", &r.W[c][k], g.W[c][k])
		check("U", &r.U[k], g.U[k])
	}
	for c := range r.A {
		check("B", &r.B[c], g.B[c])
	}
	check("C", &r.C, g.C)
}

func TestUntrainedGreedyIsLegal(t *testing.T) {
	for _, tree := range trees(t) {
		r := New(tree, featureset.Board(tree))
		for row := range tree.Decisions {
			if c := r.Greedy(row); tree.Next[row][c] < 0 {
				t.Fatalf("%s row %d: greedy picked illegal action %d", tree.Name, row, c)
			}
		}
	}
}

func TestSaveLoadRejectsOtherGame(t *testing.T) {
	ts := trees(t)
	path := t.TempDir() + "/readout.bin"
	if err := New(ts[0], featureset.Board(ts[0])).Save(path); err != nil {
		t.Fatal(err)
	}
	if err := New(ts[0], featureset.Board(ts[0])).Load(path); err != nil {
		t.Fatalf("same game: %v", err)
	}
	// Hexapawn 3x4 has 24 observation bits and 36 actions, tic-tac-toe 18 and 9.
	if err := New(ts[1], featureset.Board(ts[1])).Load(path); err == nil {
		t.Fatal("loaded a tic-tac-toe readout into hexapawn")
	}
}
