// Package hexapawn implements Hexapawn (Martin Gardner, 1962) on a board of any size.
//
// Each player starts with a row of pawns on their home row. A pawn moves one cell straight
// forward onto an empty cell, or one cell diagonally forward to capture an enemy pawn. You win
// by reaching the far row, or when the opponent has no legal move (which includes having no
// pawns left). There are no draws.
//
// A position is seen from the side to move: cells are numbered row*Cols + col, and the mover's
// pawns always advance towards higher rows. Play rotates the board 180° and swaps sides, so
// both players share one encoding, exactly what the fly sees.
package hexapawn

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/empire/fruit-fly/internal/game"
)

// Directions a pawn can move, from the mover's point of view.
const (
	Forward = iota
	CaptureLeft
	CaptureRight
)

// Pos is a position from the perspective of the side to move (bit i = cell i).
type Pos struct{ Mine, Opp uint64 }

// Game is Hexapawn on a Cols x Rows board. Action = from*3 + direction.
type Game struct{ Cols, Rows int }

var _ game.Game[Pos] = Game{}

// New returns Hexapawn on a board cols wide and rows tall.
func New(cols, rows int) (Game, error) {
	switch {
	case cols < 1 || cols > 26:
		return Game{}, fmt.Errorf("hexapawn: %d columns, want 1..26", cols)
	case rows < 3:
		return Game{}, fmt.Errorf("hexapawn: %d rows, want at least 3", rows)
	case cols*rows > 32:
		return Game{}, fmt.Errorf("hexapawn: %dx%d has %d cells, at most 32 fit a position key", cols, rows, cols*rows)
	}
	return Game{Cols: cols, Rows: rows}, nil
}

func (g Game) Name() string    { return fmt.Sprintf("hexapawn-%dx%d", g.Cols, g.Rows) }
func (g Game) cells() int      { return g.Cols * g.Rows }
func (g Game) NumActions() int { return 3 * g.cells() }

// Start puts every pawn on its home row: row 0 for the mover, the last row for the opponent.
func (g Game) Start() Pos {
	var p Pos
	for c := range g.Cols {
		p.Mine |= 1 << c
		p.Opp |= 1 << ((g.Rows-1)*g.Cols + c)
	}
	return p
}

func (g Game) Key(p Pos) uint64 { return p.Mine | p.Opp<<g.cells() }

// target returns the cell a pawn on from reaches in direction dir, and whether that move is legal.
func (g Game) target(p Pos, from, dir int) (int, bool) {
	r, c := from/g.Cols, from%g.Cols
	if p.Mine>>from&1 == 0 || r+1 >= g.Rows {
		return 0, false
	}
	switch dir {
	case Forward:
		to := from + g.Cols
		return to, (p.Mine|p.Opp)>>to&1 == 0
	case CaptureLeft:
		to := from + g.Cols - 1
		return to, c > 0 && p.Opp>>to&1 == 1
	case CaptureRight:
		to := from + g.Cols + 1
		return to, c < g.Cols-1 && p.Opp>>to&1 == 1
	}
	return 0, false
}

func (g Game) Legal(p Pos, dst []int) []int {
	for from := range g.cells() {
		for dir := range 3 {
			if _, ok := g.target(p, from, dir); ok {
				dst = append(dst, from*3+dir)
			}
		}
	}
	return dst
}

// Outcome: the side to move has lost if an enemy pawn stands on its home row or it cannot move.
func (g Game) Outcome(p Pos) (int, bool) {
	if p.Opp&(1<<g.Cols-1) != 0 || len(g.Legal(p, nil)) == 0 {
		return -1, true
	}
	return 0, false
}

func (g Game) Play(p Pos, action int) Pos {
	from := action / 3
	to, _ := g.target(p, from, action%3)
	mine := p.Mine&^(1<<from) | 1<<to
	opp := p.Opp &^ (1 << to)
	return Pos{Mine: g.rotate(opp), Opp: g.rotate(mine)}
}

// rotate turns a mask 180°: cell i becomes cells-1-i.
func (g Game) rotate(mask uint64) uint64 {
	var out uint64
	for i := range g.cells() {
		if mask>>i&1 == 1 {
			out |= 1 << (g.cells() - 1 - i)
		}
	}
	return out
}

// Layout: channel 0 is the mover's pawns, channel 1 the opponent's.
func (g Game) Layout() game.Layout { return game.Layout{Channels: 2, Rows: g.Rows, Cols: g.Cols} }

func (g Game) Observe(p Pos, obs []bool) {
	for i := range g.cells() {
		obs[i] = p.Mine>>i&1 == 1
		obs[g.cells()+i] = p.Opp>>i&1 == 1
	}
}

// ---- terminal UI: the first player (W) sits at the bottom, squares are named like a1 --------

const (
	bold  = "\033[1m"
	red   = "\033[31m"
	cyan  = "\033[36m"
	reset = "\033[0m"
)

// absolute returns the first player's and second player's pawns in the first player's view,
// and maps a mover cell to that view.
func (g Game) absolute(p Pos, firstToMove bool) (first, second uint64, cell func(int) int) {
	if firstToMove {
		return p.Mine, p.Opp, func(i int) int { return i }
	}
	return g.rotate(p.Opp), g.rotate(p.Mine), func(i int) int { return g.cells() - 1 - i }
}

func (g Game) square(cell int) string {
	return string(rune('a'+cell%g.Cols)) + strconv.Itoa(cell/g.Cols+1)
}

func (g Game) Render(p Pos, firstToMove bool) []string {
	first, second, _ := g.absolute(p, firstToMove)
	var rows []string
	for r := g.Rows - 1; r >= 0; r-- {
		row := fmt.Sprintf("%2d |", r+1)
		for c := range g.Cols {
			i := r*g.Cols + c
			switch {
			case first>>i&1 == 1:
				row += " " + bold + red + "W" + reset
			case second>>i&1 == 1:
				row += " " + bold + cyan + "B" + reset
			default:
				row += " ."
			}
		}
		rows = append(rows, row)
	}
	files := "    "
	for c := range g.Cols {
		files += " " + string(rune('a'+c))
	}
	return append(rows, files)
}

// ParseAction reads a move like "b1b2" (forward) or "b2c3" (capture).
func (g Game) ParseAction(p Pos, firstToMove bool, text string) (int, bool) {
	from, rest, ok := g.parseSquare(strings.ToLower(text))
	if !ok {
		return 0, false
	}
	to, rest, ok := g.parseSquare(rest)
	if !ok || rest != "" {
		return 0, false
	}
	_, _, toMover := g.absolute(p, firstToMove) // the mapping is its own inverse
	from, to = toMover(from), toMover(to)
	for dir := range 3 {
		if t, legal := g.target(p, from, dir); legal && t == to {
			return from*3 + dir, true
		}
	}
	return 0, false
}

func (g Game) parseSquare(s string) (cell int, rest string, ok bool) {
	if len(s) < 2 || s[0] < 'a' || int(s[0]-'a') >= g.Cols {
		return 0, s, false
	}
	n := 1
	for n < len(s) && s[n] >= '0' && s[n] <= '9' {
		n++
	}
	row, err := strconv.Atoi(s[1:n])
	if err != nil || row < 1 || row > g.Rows {
		return 0, s, false
	}
	return (row-1)*g.Cols + int(s[0]-'a'), s[n:], true
}

func (g Game) ActionName(p Pos, firstToMove bool, action int) string {
	from := action / 3
	to, _ := g.target(p, from, action%3)
	_, _, abs := g.absolute(p, firstToMove)
	return g.square(abs(from)) + g.square(abs(to))
}

func (g Game) InputHint() string { return "e.g. a1a2, or b2c3 to capture" }
