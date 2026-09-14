// Package readout is the only part that learns: a linear map from features to a move.
//
//	scores = W · standardize(features[position]) + b     // one score per action
//	illegal actions -> -Inf, softmax -> probabilities, sample (training) or argmax (playing)
//
// Training is REINFORCE: play a batch of games, then make each move the learner made more
// likely in proportion to (final result − predicted result). The prediction comes from a
// second linear head (the value baseline), which reduces noise. Gradients are written by
// hand, because the model is small enough to derive them on paper (see Accumulate).
//
// Nothing here knows which game is played: positions are rows of a game.Tree, and the same
// code trains a readout on any feature set (see package featureset).
package readout

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/empire/fruit-fly/internal/game"
)

// Readout is a linear policy head (one per action) and a linear value head on standardized features.
type Readout struct {
	Tree *game.Tree
	X    [][]float64 // standardized features, [row][D]
	D    int         // features
	A    int         // actions

	W [][]float64 // [action][D] policy weights
	B []float64   // [action] policy bias
	U []float64   // value weights
	C float64     // value bias
}

// New creates a readout for t on features x ([row][D]) that starts as a uniformly random
// player (all weights zero).
func New(t *game.Tree, x [][]float64) *Readout {
	r := &Readout{Tree: t, X: Standardize(x), D: len(x[0]), A: t.NumActions, U: make([]float64, len(x[0]))}
	r.W, r.B = make([][]float64, r.A), make([]float64, r.A)
	for c := range r.A {
		r.W[c] = make([]float64, r.D)
	}
	return r
}

func dot(a, b []float64) float64 {
	s := 0.0
	for k := range a {
		s += a[k] * b[k]
	}
	return s
}

// Probs returns move probabilities for a position; illegal actions get 0.
func (r *Readout) Probs(row int) []float64 {
	x, next := r.X[row], r.Tree.Next[row]
	z := make([]float64, r.A)
	top := math.Inf(-1)
	for c := range r.A {
		if next[c] >= 0 {
			z[c] = dot(r.W[c], x) + r.B[c]
			top = max(top, z[c])
		}
	}
	// softmax, shifted by the largest score so exp() can't overflow
	p := make([]float64, r.A)
	sum := 0.0
	for c := range r.A {
		if next[c] >= 0 {
			p[c] = math.Exp(z[c] - top)
			sum += p[c]
		}
	}
	for c := range p {
		p[c] /= sum
	}
	return p
}

// Value predicts the result (+1 win … -1 loss) for the side to move.
func (r *Readout) Value(row int) float64 { return dot(r.U, r.X[row]) + r.C }

// Greedy returns the most likely legal action.
func (r *Readout) Greedy(row int) int {
	p := r.Probs(row)
	best := -1
	for c := range r.A {
		if r.Tree.Next[row][c] >= 0 && (best < 0 || p[c] > p[best]) {
			best = c
		}
	}
	return best
}

// Sample is one move the learner made, with what it knew at the time.
type Sample struct {
	Row    int
	Action int
	Return float64   // final result from the mover's point of view
	Probs  []float64 // policy when the move was chosen
	Value  float64   // value prediction when the move was chosen
}

// Grad has the same shape as the readout's parameters.
type Grad struct {
	W [][]float64
	B []float64
	U []float64
	C float64
}

func newGrad(a, d int) *Grad {
	g := &Grad{W: make([][]float64, a), B: make([]float64, a), U: make([]float64, d)}
	for c := range a {
		g.W[c] = make([]float64, d)
	}
	return g
}

// zero clears g for reuse.
func (g *Grad) zero() {
	for c := range g.W {
		clear(g.W[c])
	}
	clear(g.B)
	clear(g.U)
	g.C = 0
}

// Accumulate adds one sample's gradient of the loss
//
//	L = −adv·log p[a]  +  ½(ret − v)²  −  β·H(p)      with adv = ret − v held constant
//
// Derivation, with z the scores and p = softmax(z):
//
//	∂(−adv·log p[a])/∂z[k] = −adv·(1[k=a] − p[k])
//	∂(−β·H)/∂z[k]          =  β·p[k]·(log p[k] + H)       where H = −Σ p log p
//	∂(½(ret − v)²)/∂v      = −(ret − v)
//
// and z[k] = W[k]·x + B[k], v = U·x + C, so each weight's gradient is that times x.
// The finite-difference test in policy_test.go checks this derivation numerically.
func (r *Readout) Accumulate(g *Grad, s Sample, entropy float64) {
	x := r.X[s.Row]
	adv := s.Return - s.Value
	h := 0.0
	for _, p := range s.Probs {
		if p > 0 {
			h -= p * math.Log(p)
		}
	}
	for c := range r.A {
		p := s.Probs[c]
		if p == 0 {
			continue // illegal action: its score never mattered
		}
		onehot := 0.0
		if c == s.Action {
			onehot = 1
		}
		dz := -adv*(onehot-p) + entropy*p*(math.Log(p)+h)
		for k, v := range x {
			g.W[c][k] += dz * v
		}
		g.B[c] += dz
	}
	dv := -(s.Return - s.Value)
	for k, v := range x {
		g.U[k] += dv * v
	}
	g.C += dv
}

// adam is the Adam optimizer: per-parameter step sizes from running gradient averages.
type adam struct {
	lr, beta1, beta2 float64
	t                int
	m, v             *Grad
}

func newAdam(a, d int, lr float64) *adam {
	return &adam{lr: lr, beta1: 0.9, beta2: 0.999, m: newGrad(a, d), v: newGrad(a, d)}
}

func (a *adam) step(r *Readout, g *Grad, scale float64) {
	a.t++
	c1 := 1 - math.Pow(a.beta1, float64(a.t))
	c2 := 1 - math.Pow(a.beta2, float64(a.t))
	update := func(param, grad, m, v *float64) {
		gr := *grad * scale
		*m = a.beta1**m + (1-a.beta1)*gr
		*v = a.beta2**v + (1-a.beta2)*gr*gr
		*param -= a.lr * (*m / c1) / (math.Sqrt(*v/c2) + 1e-8)
	}
	for c := range r.A {
		for k := range r.W[c] {
			update(&r.W[c][k], &g.W[c][k], &a.m.W[c][k], &a.v.W[c][k])
		}
		update(&r.B[c], &g.B[c], &a.m.B[c], &a.v.B[c])
	}
	for k := range r.U {
		update(&r.U[k], &g.U[k], &a.m.U[k], &a.v.U[k])
	}
	update(&r.C, &g.C, &a.m.C, &a.v.C)
}

const readoutMagic = "FLYREAD1"

// Save writes the policy weights (the value head is only needed during training).
func (r *Readout) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	put := func(v any) {
		if err == nil {
			err = binary.Write(w, binary.LittleEndian, v)
		}
	}
	put([]byte(readoutMagic))
	put(int64(r.D))
	for c := range r.A {
		put(r.W[c])
	}
	put(r.B)
	if err == nil {
		err = w.Flush()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}

// Load reads policy weights written by Save into a readout built on the same game and features.
func (r *Readout) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("%w (run `fly train -game %s` first)", err, r.Tree.Name)
	}
	defer f.Close()
	rd := bufio.NewReader(f)
	head := make([]byte, len(readoutMagic))
	if _, err := io.ReadFull(rd, head); err != nil || string(head) != readoutMagic {
		return fmt.Errorf("%s: bad header", path)
	}
	var d int64
	get := func(v any) {
		if err == nil {
			err = binary.Read(rd, binary.LittleEndian, v)
		}
	}
	get(&d)
	if err == nil && int(d) != r.D {
		return fmt.Errorf("%s: %d features, want %d", path, d, r.D)
	}
	for c := range r.A {
		get(r.W[c])
	}
	get(r.B)
	if err == nil {
		if _, extra := rd.ReadByte(); extra != io.EOF {
			err = fmt.Errorf("%s: longer than %d actions x %d features (trained for another game?)", path, r.A, r.D)
		}
	}
	return err
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
