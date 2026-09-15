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
	mean []float64
	std  []float64
	enc  func([]bool) []float64 // optional: features from any observation

	W [][]float64 // [action][D] policy weights
	B []float64   // [action] policy bias
	U []float64   // value weights
	C float64     // value bias
}

// New creates a readout for t on features x ([row][D]) that starts as a uniformly random
// player (all weights zero).
func New(t *game.Tree, x [][]float64) *Readout {
	out, mean, std := scale(x)
	r := &Readout{Tree: t, X: out, D: len(x[0]), A: t.NumActions, mean: mean, std: std, U: make([]float64, len(x[0]))}
	r.W, r.B = make([][]float64, r.A), make([]float64, r.A)
	for c := range r.A {
		r.W[c] = make([]float64, r.D)
	}
	return r
}

// AttachEncoder lets the readout score positions that are not interned, using the same
// scaler as the training cache. Brain features have no encoder.
func (r *Readout) AttachEncoder(enc func([]bool) []float64) { r.enc = enc }

func (r *Readout) scaled(obs []bool) []float64 {
	return applyScale(r.enc(obs), r.mean, r.std)
}

func dot(a, b []float64) float64 {
	s := 0.0
	for k := range a {
		s += a[k] * b[k]
	}
	return s
}

// ProbsOn scores a feature vector with an explicit legal set (used for live play).
func (r *Readout) ProbsOn(x []float64, legal []int) []float64 {
	z := make([]float64, r.A)
	top := math.Inf(-1)
	for _, c := range legal {
		z[c] = dot(r.W[c], x) + r.B[c]
		top = max(top, z[c])
	}
	p := make([]float64, r.A)
	sum := 0.0
	for _, c := range legal {
		p[c] = math.Exp(z[c] - top)
		sum += p[c]
	}
	for c := range p {
		p[c] /= sum
	}
	return p
}

func uniform(legal []int, a int) []float64 {
	p := make([]float64, a)
	if len(legal) == 0 {
		return p
	}
	u := 1 / float64(len(legal))
	for _, c := range legal {
		p[c] = u
	}
	return p
}

func greedyOf(p []float64, legal []int) int {
	best := legal[0]
	for _, c := range legal[1:] {
		if p[c] > p[best] {
			best = c
		}
	}
	return best
}

// Probs returns move probabilities for an interned position; illegal actions get 0.
func (r *Readout) Probs(row int) []float64 {
	return r.ProbsOn(r.X[row], r.Tree.LegalActions(row))
}

// ProbsWalk scores the current position of a game, including sample misses.
func (r *Readout) ProbsWalk(w *game.Walk) []float64 {
	if w.Row >= 0 {
		return r.Probs(w.Row)
	}
	if r.enc == nil {
		return uniform(w.Legal, r.A)
	}
	return r.ProbsOn(r.scaled(w.Obs()), w.Legal)
}

// Value predicts the result (+1 win … -1 loss) for the side to move.
func (r *Readout) Value(row int) float64 { return dot(r.U, r.X[row]) + r.C }

// ValueWalk is Value at an interned row, or the encoder's value off-tree.
func (r *Readout) ValueWalk(w *game.Walk) float64 {
	if w.Row >= 0 {
		return r.Value(w.Row)
	}
	if r.enc == nil {
		return 0
	}
	return dot(r.U, r.scaled(w.Obs())) + r.C
}

// Greedy returns the most likely legal action of an interned row.
func (r *Readout) Greedy(row int) int {
	legal := r.Tree.LegalActions(row)
	return greedyOf(r.Probs(row), legal)
}

// GreedyWalk is Greedy at an interned row, or the encoder / uniform off-tree.
func (r *Readout) GreedyWalk(w *game.Walk) int {
	return greedyOf(r.ProbsWalk(w), w.Legal)
}

// Sample is one move the learner made, with what it knew at the time.
type Sample struct {
	Row    int
	Action int
	Return float64   // final result from the mover's point of view
	Probs  []float64 // policy when the move was chosen
	Value  float64   // value prediction when the move was chosen
	X      []float64 // set when Row < 0 (off-tree features)
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
	x := s.X
	if x == nil {
		x = r.X[s.Row]
	}
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
		return err
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
	out, _, _ := scale(x)
	return out
}

func scale(x [][]float64) (out [][]float64, mean, std []float64) {
	d := len(x[0])
	mean, std = make([]float64, d), make([]float64, d)
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
	out = make([][]float64, len(x))
	for i, row := range x {
		out[i] = applyScale(row, mean, std)
	}
	return out, mean, std
}

func applyScale(row, mean, std []float64) []float64 {
	out := make([]float64, len(row))
	for k, v := range row {
		out[k] = (v - mean[k]) / std[k]
	}
	return out
}

// ScaleCounts standardizes live spike counts with the readout's training scaler.
func (r *Readout) ScaleCounts(counts []uint8) []float64 {
	raw := make([]float64, len(counts))
	for i, v := range counts {
		raw[i] = float64(v)
	}
	return applyScale(raw, r.mean, r.std)
}
