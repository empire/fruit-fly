package game

import (
	"cmp"
	"fmt"
	"math/rand/v2"
	"slices"
)

// Tree is a set of interned positions plus the rules, so the same tables work for a
// complete enumeration (tic-tac-toe) and a small random sample (8x8 Othello).
//
// Nodes 0..Decisions-1 are positions where someone must move ("rows", sorted by Key); the
// remaining interned nodes are finished games. Play uses Walk, which follows the rules
// even when a position is not in the tree.
type Tree struct {
	Name       string
	Layout     Layout
	NumActions int
	Decisions  int       // interned positions that need a move
	Root       int       // the starting position
	Partial    bool      // true if this is a sample, not every reachable position
	Next       [][]int32 // [row][action] -> interned node, -1 if illegal or unseen
	Value      []int8    // [node] result for interned terminals; decisions if solved
	Best       [][]bool  // [row][action] optimal; nil when Partial
	Obs        [][]bool  // [row] what the fly sees

	obsIndex map[string]int32
	legal    func(row int) []int
	newWalk  func() *Walk
	ui       ui
}

// ui holds the game's text functions, bound to interned states.
type ui struct {
	render     func(node int, firstToMove bool) []string
	parse      func(node int, firstToMove bool, text string) (int, bool)
	actionName func(node int, firstToMove bool, action int) string
	probBoard  func(node int, probs []float64) []string // nil if the game has none
	hint       string
}

// Compile enumerates every reachable position of g and solves it with minimax.
func Compile[S comparable](g Game[S]) (*Tree, error) {
	var decisions, terminals []S
	seen := map[S]bool{}
	stack := []S{g.Start()}
	var legal []int
	for len(stack) > 0 {
		s := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[s] {
			continue
		}
		seen[s] = true
		if _, over := g.Outcome(s); over {
			terminals = append(terminals, s)
			continue
		}
		decisions = append(decisions, s)
		legal = g.Legal(s, legal[:0])
		if len(legal) == 0 {
			return nil, fmt.Errorf("%s: a position that is not over has no legal moves", g.Name())
		}
		for _, a := range legal {
			stack = append(stack, g.Play(s, a))
		}
	}
	return finish(g, decisions, terminals, false)
}

// Sample builds a tree from at most limit unique decision positions, found by playing
// random games. Use this when the game is too large to enumerate. The rules still play
// the full game; only the interned rows get a brain cache and a perfect-play label is
// not computed.
func Sample[S comparable](g Game[S], limit int, seed uint64) (*Tree, error) {
	if limit < 1 {
		return nil, fmt.Errorf("%s: sample limit %d, want at least 1", g.Name(), limit)
	}
	rng := rand.New(rand.NewPCG(seed, 1))
	seen := map[S]bool{}
	var decisions, terminals []S
	var legal []int

	add := func(s S) {
		if seen[s] {
			return
		}
		if _, over := g.Outcome(s); over {
			seen[s] = true
			terminals = append(terminals, s)
			return
		}
		if len(decisions) >= limit {
			return
		}
		seen[s] = true
		decisions = append(decisions, s)
	}

	add(g.Start())
	stuck := 0
	for len(decisions) < limit {
		before := len(decisions)
		s := g.Start()
		for {
			add(s)
			if _, over := g.Outcome(s); over {
				break
			}
			legal = g.Legal(s, legal[:0])
			if len(legal) == 0 {
				return nil, fmt.Errorf("%s: a position that is not over has no legal moves", g.Name())
			}
			s = g.Play(s, legal[rng.IntN(len(legal))])
		}
		if len(decisions) == before {
			stuck++
			if stuck > 32 {
				break
			}
		} else {
			stuck = 0
		}
	}
	return finish(g, decisions, terminals, true)
}

func finish[S comparable](g Game[S], decisions, terminals []S, partial bool) (*Tree, error) {
	byKey := func(a, b S) int {
		if c := cmp.Compare(g.Key(a), g.Key(b)); c != 0 {
			return c
		}
		if a == b {
			return 0
		}
		return cmp.Compare(fmt.Sprintf("%#v", a), fmt.Sprintf("%#v", b))
	}
	slices.SortFunc(decisions, byKey)
	slices.SortFunc(terminals, byKey)
	states := append(decisions, terminals...)
	index := make(map[S]int32, len(states))
	for i, s := range states {
		if !partial && i > 0 && g.Key(states[i-1]) == g.Key(s) && i != len(decisions) {
			return nil, fmt.Errorf("%s: two states share key %d", g.Name(), g.Key(s))
		}
		index[s] = int32(i)
	}

	layout := g.Layout()
	t := &Tree{Name: g.Name(), Layout: layout, NumActions: g.NumActions(), Decisions: len(decisions),
		Partial: partial, Root: int(index[g.Start()]), Next: make([][]int32, len(decisions)),
		Value: make([]int8, len(states)), Obs: make([][]bool, len(decisions)),
		obsIndex: make(map[string]int32, len(decisions))}
	if !partial {
		t.Best = make([][]bool, len(decisions))
	}

	var legal []int
	for row, s := range decisions {
		t.Next[row] = make([]int32, t.NumActions)
		for a := range t.Next[row] {
			t.Next[row][a] = -1
		}
		for _, a := range g.Legal(s, legal[:0]) {
			if a < 0 || a >= t.NumActions {
				return nil, fmt.Errorf("%s: action %d outside 0..%d", g.Name(), a, t.NumActions-1)
			}
			if next, ok := index[g.Play(s, a)]; ok {
				t.Next[row][a] = next
			}
		}
		t.Obs[row] = make([]bool, layout.Size())
		g.Observe(s, t.Obs[row])
		key := obsKey(t.Obs[row])
		if _, dup := t.obsIndex[key]; !dup {
			t.obsIndex[key] = int32(row)
		}
	}

	for i, s := range terminals {
		v, _ := g.Outcome(s)
		t.Value[len(decisions)+i] = int8(v)
	}
	if !partial {
		t.solve()
	}

	t.legal = func(row int) []int {
		if row < 0 || row >= t.Decisions {
			return nil
		}
		return g.Legal(states[row], nil)
	}
	t.newWalk = func() *Walk { return newWalk(t, g, index) }
	t.ui = ui{
		render: func(n int, first bool) []string { return g.Render(states[n], first) },
		parse:  func(n int, first bool, text string) (int, bool) { return g.ParseAction(states[n], first, text) },
		actionName: func(n int, first bool, a int) string {
			return g.ActionName(states[n], first, a)
		},
		hint: g.InputHint(),
	}
	if pb, ok := g.(ProbabilityBoard[S]); ok {
		t.ui.probBoard = func(n int, probs []float64) []string { return pb.ProbabilityBoard(states[n], probs) }
	}
	return t, nil
}

func newWalk[S comparable](t *Tree, g Game[S], index map[S]int32) *Walk {
	s := g.Start()
	w := &Walk{Tree: t, First: true}
	obs := make([]bool, t.Layout.Size())
	sync := func() {
		if i, ok := index[s]; ok {
			w.Row = int(i)
		} else {
			w.Row = -1
		}
		v, over := g.Outcome(s)
		w.Value, w.Over = v, over
		w.Legal = nil
		if !over {
			w.Legal = g.Legal(s, nil)
		}
		clear(obs)
		g.Observe(s, obs)
		w.obs = slices.Clone(obs)
		w.render = g.Render(s, w.First)
		first := w.First
		cur := s
		w.parse = func(text string) (int, bool) { return g.ParseAction(cur, first, text) }
		w.name = func(a int) string { return g.ActionName(cur, first, a) }
		if pb, ok := g.(ProbabilityBoard[S]); ok {
			w.probs = func(p []float64) []string { return pb.ProbabilityBoard(cur, p) }
		}
	}
	w.step = func(action int) {
		s = g.Play(s, action)
		w.First = !w.First
		sync()
	}
	sync()
	return w
}

func obsKey(obs []bool) string {
	b := make([]byte, len(obs))
	for i, v := range obs {
		if v {
			b[i] = 1
		}
	}
	return string(b)
}

// Nodes is the number of interned positions, finished games included.
func (t *Tree) Nodes() int { return len(t.Value) }

// IsTerminal reports whether an interned node is a finished game.
func (t *Tree) IsTerminal(node int) bool { return node >= t.Decisions }

// LegalActions returns the legal actions of an interned decision row, from the rules.
func (t *Tree) LegalActions(row int) []int {
	if t.legal != nil {
		return t.legal(row)
	}
	var actions []int
	for a, next := range t.Next[row] {
		if next >= 0 {
			actions = append(actions, a)
		}
	}
	return actions
}

// RowOfObservation finds an interned decision position that looks exactly like obs.
func (t *Tree) RowOfObservation(obs []bool) (int, bool) {
	row, ok := t.obsIndex[obsKey(obs)]
	return int(row), ok
}

// Render draws an interned node; firstToMove says whether the player who started is to move.
func (t *Tree) Render(node int, firstToMove bool) []string { return t.ui.render(node, firstToMove) }

// ParseAction reads a human move typed at an interned decision position.
func (t *Tree) ParseAction(row int, firstToMove bool, text string) (int, bool) {
	a, ok := t.ui.parse(row, firstToMove, text)
	if !ok || a < 0 || a >= t.NumActions {
		return 0, false
	}
	for _, legal := range t.LegalActions(row) {
		if legal == a {
			return a, true
		}
	}
	return 0, false
}

// ActionName writes an action the way a human would type it.
func (t *Tree) ActionName(row int, firstToMove bool, action int) string {
	return t.ui.actionName(row, firstToMove, action)
}

// InputHint describes the move syntax, e.g. "1-9".
func (t *Tree) InputHint() string { return t.ui.hint }

// ProbabilityBoard draws move probabilities on the board, if the game supports it.
func (t *Tree) ProbabilityBoard(row int, probs []float64) ([]string, bool) {
	if t.ui.probBoard == nil {
		return nil, false
	}
	return t.ui.probBoard(row, probs), true
}
