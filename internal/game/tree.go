package game

import (
	"cmp"
	"fmt"
	"slices"
)

// Tree is a compiled game: every reachable position, numbered, with its moves and results.
//
// Nodes 0..Decisions-1 are positions where someone must move ("rows", sorted by Key); the
// remaining nodes are finished games. Everything that learns or plays indexes these tables
// instead of calling the rules, so it works for any game.
type Tree struct {
	Name       string
	Layout     Layout
	NumActions int
	Decisions  int       // number of positions that need a move
	Root       int       // the starting position
	Next       [][]int32 // [row][action] -> node after the move, -1 if illegal
	Value      []int8    // [node] result for the side to move under perfect play (-1, 0, +1)
	Best       [][]bool  // [row][action] the move achieves Value
	Obs        [][]bool  // [row] what the fly sees

	obsIndex map[string]int32
	ui       ui
}

// ui holds the game's text functions, bound to the tree's states.
type ui struct {
	render     func(node int, firstToMove bool) []string
	parse      func(node int, firstToMove bool, text string) (int, bool)
	actionName func(node int, firstToMove bool, action int) string
	probBoard  func(node int, probs []float64) []string // nil if the game has none
	hint       string
}

// Compile enumerates every reachable position of g and solves it with minimax.
func Compile[S comparable](g Game[S]) (*Tree, error) {
	// 1. Every reachable state, split into decisions and finished games.
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

	// 2. Number them: decisions by key, then finished games by key.
	byKey := func(a, b S) int { return cmp.Compare(g.Key(a), g.Key(b)) }
	slices.SortFunc(decisions, byKey)
	slices.SortFunc(terminals, byKey)
	states := append(decisions, terminals...)
	index := make(map[S]int32, len(states))
	for i, s := range states {
		if i > 0 && g.Key(states[i-1]) == g.Key(s) && i != len(decisions) {
			return nil, fmt.Errorf("%s: two states share key %d", g.Name(), g.Key(s))
		}
		index[s] = int32(i)
	}

	layout := g.Layout()
	t := &Tree{Name: g.Name(), Layout: layout, NumActions: g.NumActions(), Decisions: len(decisions),
		Root: int(index[g.Start()]), Next: make([][]int32, len(decisions)),
		Value: make([]int8, len(states)), Best: make([][]bool, len(decisions)),
		Obs: make([][]bool, len(decisions)), obsIndex: make(map[string]int32, len(decisions))}

	// 3. Moves and observations.
	for row, s := range decisions {
		t.Next[row] = make([]int32, t.NumActions)
		for a := range t.Next[row] {
			t.Next[row][a] = -1
		}
		for _, a := range g.Legal(s, legal[:0]) {
			if a < 0 || a >= t.NumActions {
				return nil, fmt.Errorf("%s: action %d outside 0..%d", g.Name(), a, t.NumActions-1)
			}
			t.Next[row][a] = index[g.Play(s, a)]
		}
		t.Obs[row] = make([]bool, layout.Size())
		g.Observe(s, t.Obs[row])
		key := obsKey(t.Obs[row])
		if _, dup := t.obsIndex[key]; !dup {
			t.obsIndex[key] = int32(row)
		}
	}

	// 4. Results under perfect play.
	for i, s := range terminals {
		v, _ := g.Outcome(s)
		t.Value[len(decisions)+i] = int8(v)
	}
	t.solve()

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

func obsKey(obs []bool) string {
	b := make([]byte, len(obs))
	for i, v := range obs {
		if v {
			b[i] = 1
		}
	}
	return string(b)
}

// Nodes is the number of positions, finished games included.
func (t *Tree) Nodes() int { return len(t.Value) }

// IsTerminal reports whether a node is a finished game.
func (t *Tree) IsTerminal(node int) bool { return node >= t.Decisions }

// LegalActions returns the legal actions of a row in increasing order.
func (t *Tree) LegalActions(row int) []int {
	var actions []int
	for a, next := range t.Next[row] {
		if next >= 0 {
			actions = append(actions, a)
		}
	}
	return actions
}

// RowOfObservation finds a decision position that looks exactly like obs.
func (t *Tree) RowOfObservation(obs []bool) (int, bool) {
	row, ok := t.obsIndex[obsKey(obs)]
	return int(row), ok
}

// Render draws a node; firstToMove says whether the player who started is to move.
func (t *Tree) Render(node int, firstToMove bool) []string { return t.ui.render(node, firstToMove) }

// ParseAction reads a human move typed at a decision position.
func (t *Tree) ParseAction(row int, firstToMove bool, text string) (int, bool) {
	a, ok := t.ui.parse(row, firstToMove, text)
	return a, ok && a >= 0 && a < t.NumActions && t.Next[row][a] >= 0
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
