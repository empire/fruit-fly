// Package tictactoe implements tic-tac-toe as a game.Game.
//
// A position is always seen from the side to move: Mine and Opp are 9-bit masks
// (bit i = cell i, cells numbered 0..8 left-to-right, top-to-bottom). This is the same
// view the fly gets, so X-to-move and O-to-move positions share one encoding. They never
// collide, because the piece counts differ.
package tictactoe

import (
	"fmt"
	"strconv"

	"github.com/empire/fruit-fly/internal/game"
)

const (
	bold  = "\033[1m"
	red   = "\033[31m"
	cyan  = "\033[36m"
	reset = "\033[0m"
)

// Full is the mask with all nine cells occupied.
const Full = 1<<9 - 1

// Lines are the eight winning lines as bitmasks.
var Lines = [8]uint16{
	0b000000111, 0b000111000, 0b111000000, // rows
	0b001001001, 0b010010010, 0b100100100, // columns
	0b100010001, 0b001010100, // diagonals
}

// Pos is a position from the perspective of the side to move.
type Pos struct{ Mine, Opp uint16 }

// Won reports whether a 9-bit mask contains a complete line.
func Won(mask uint16) bool {
	for _, l := range Lines {
		if mask&l == l {
			return true
		}
	}
	return false
}

// Game is tic-tac-toe. Actions are cells 0..8.
type Game struct{}

var _ game.Game[Pos] = Game{}

// New returns tic-tac-toe.
func New() Game { return Game{} }

func (Game) Name() string    { return "tictactoe" }
func (Game) Start() Pos      { return Pos{} }
func (Game) NumActions() int { return 9 }

// Key packs a position into 18 bits: mine | opp<<9.
func (Game) Key(p Pos) uint64 { return uint64(p.Mine) | uint64(p.Opp)<<9 }

// Legal appends the empty cells in increasing order.
func (Game) Legal(p Pos, dst []int) []int {
	taken := p.Mine | p.Opp
	for c := range 9 {
		if taken>>c&1 == 0 {
			dst = append(dst, c)
		}
	}
	return dst
}

// Outcome returns the result for the side to move (-1 loss, 0 draw) and whether the game is over.
// The side to move can never have just won, so a win is impossible here.
func (Game) Outcome(p Pos) (int, bool) {
	if Won(p.Opp) {
		return -1, true
	}
	if p.Mine|p.Opp == Full {
		return 0, true
	}
	return 0, false
}

// Play places a piece for the side to move and hands the turn over (the perspective flips).
func (Game) Play(p Pos, cell int) Pos {
	return Pos{Mine: p.Opp, Opp: p.Mine | 1<<cell}
}

// Layout: channel 0 is the mover's pieces, channel 1 the opponent's, on a 3x3 board.
func (Game) Layout() game.Layout { return game.Layout{Channels: 2, Rows: 3, Cols: 3} }

// Observe lights the cells holding my pieces (channel 0) and the opponent's (channel 1).
func (Game) Observe(p Pos, obs []bool) {
	for c := range 9 {
		obs[c] = p.Mine>>c&1 == 1
		obs[9+c] = p.Opp>>c&1 == 1
	}
}

// Render returns three text rows; empty cells show their 1-9 number.
func (Game) Render(p Pos, xToMove bool) []string {
	xs, os := p.Mine, p.Opp
	if !xToMove {
		xs, os = os, xs
	}
	var rows []string
	for r := range 3 {
		row := ""
		for c := r * 3; c < r*3+3; c++ {
			cell := strconv.Itoa(c + 1)
			switch {
			case xs>>c&1 == 1:
				cell = bold + red + "X" + reset
			case os>>c&1 == 1:
				cell = bold + cyan + "O" + reset
			}
			if c > r*3 {
				row += " | "
			}
			row += cell
		}
		rows = append(rows, " "+row)
		if r < 2 {
			rows = append(rows, "---+---+---")
		}
	}
	return rows
}

// ParseAction reads a cell number 1-9.
func (Game) ParseAction(_ Pos, _ bool, text string) (int, bool) {
	n, err := strconv.Atoi(text)
	return n - 1, err == nil && n >= 1 && n <= 9
}

func (Game) ActionName(_ Pos, _ bool, cell int) string { return strconv.Itoa(cell + 1) }
func (Game) InputHint() string                         { return "1-9" }

// ProbabilityBoard shows each empty cell's move probability in a 3x3 grid.
func (Game) ProbabilityBoard(p Pos, probs []float64) []string {
	var rows []string
	for r := range 3 {
		row := ""
		for c := r * 3; c < r*3+3; c++ {
			if (p.Mine|p.Opp)>>c&1 == 1 {
				row += "     ."
			} else {
				row += fmt.Sprintf("  %3.0f%%", 100*probs[c])
			}
		}
		rows = append(rows, row)
	}
	return rows
}
