// Package game is the part of the program that knows what a game is, and nothing about brains.
//
// A game is described once, as Rules plus how it looks (Observer) and how a human talks to it
// (TextUI). Compile then walks every reachable position and turns the game into a Tree: plain
// tables of positions, moves, results and perfect play. Everything downstream (the brain cache,
// training, evaluation, the terminal UI) works on a Tree, so it never needs to know which game
// it is playing.
//
// Supported games: two players who alternate, perfect information, and few enough positions to
// enumerate them all (the brain is simulated once per position).
package game

// Rules are the mechanics of a game. A state S is always seen from the side to move, so the
// same code plays both sides.
type Rules[S comparable] interface {
	Start() S
	// Key must be unique per state. It only orders positions, so results are reproducible.
	Key(s S) uint64
	// NumActions is the size of the action space; actions are 0..NumActions-1.
	NumActions() int
	// Legal appends the legal actions of a non-terminal state to dst, in increasing order.
	Legal(s S, dst []int) []int
	// Play makes a legal move and hands the turn over.
	Play(s S, action int) S
	// Outcome reports whether the game is over and, if so, the result for the side to move:
	// -1 loss, 0 draw, +1 win.
	Outcome(s S) (value int, over bool)
}

// Layout is the shape of what the fly sees: Channels boards of Rows x Cols cells.
type Layout struct{ Channels, Rows, Cols int }

// Cells is the number of cells on one channel.
func (l Layout) Cells() int { return l.Rows * l.Cols }

// Size is the length of an observation.
func (l Layout) Size() int { return l.Channels * l.Cells() }

// Observer turns a state into what the eyes see.
type Observer[S comparable] interface {
	Layout() Layout
	// Observe fills obs (length Layout().Size()): obs[channel*Cells + cell] is true when that
	// cell is lit on that channel. Channel 0 conventionally holds the mover's pieces.
	Observe(s S, obs []bool)
}

// TextUI lets a human play in the terminal. firstToMove says whether the side to move is the
// player who started, so the board can be drawn from a fixed point of view.
type TextUI[S comparable] interface {
	Render(s S, firstToMove bool) []string
	ParseAction(s S, firstToMove bool, text string) (action int, ok bool)
	ActionName(s S, firstToMove bool, action int) string
	InputHint() string // e.g. "1-9"
}

// ProbabilityBoard is optional: a game can draw move probabilities on its own board.
type ProbabilityBoard[S comparable] interface {
	ProbabilityBoard(s S, probs []float64) []string
}

// Game is everything needed to compile, train on and play a game.
type Game[S comparable] interface {
	Name() string
	Rules[S]
	Observer[S]
	TextUI[S]
}
