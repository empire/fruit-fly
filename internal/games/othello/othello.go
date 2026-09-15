// Package othello implements 8x8 Othello (Reversi) as a game.Game.
//
// Black places first from the World Othello Federation opening (White on d4
// and e5, Black on e4 and d5). A move lays a disc that sandwiches one or more
// opponent discs in a straight line; those discs flip. If you cannot place you
// pass, unless the opponent also cannot, in which case whoever has more discs
// wins.
//
// 8x8 is not enumerable (~10^28 positions). The registry builds a Sample of
// SampleSize random-game positions; play follows the real rules via Walk.
// A position is seen from the side to move. Cells are numbered row*8 + col
// with row 0 at the bottom (a1 = 0), the same view the fly gets. Play swaps
// the two masks and does not rotate: the board has no facing direction.
package othello

import (
	"fmt"
	"math/bits"
	"strconv"
	"strings"

	"github.com/empire/fruit-fly/internal/game"
)

const (
	Size       = 8
	N          = Size * Size
	Pass       = N // action when the side to move has no placement
	SampleSize = 128
)

const (
	bold  = "\033[1m"
	red   = "\033[31m"
	cyan  = "\033[36m"
	reset = "\033[0m"
)

var dirs = [8][2]int{
	{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1},
}

// Pos is a position from the perspective of the side to move (bit i = cell i).
type Pos struct{ Mine, Opp uint64 }

// Game is 8x8 Othello. Actions are cells 0..63, or Pass.
type Game struct{}

var _ game.Game[Pos] = Game{}

// New returns 8x8 Othello.
func New() Game { return Game{} }

func (Game) Name() string    { return "othello" }
func (Game) NumActions() int { return N + 1 }

// Start is the WOF opening: Black (to move) on e4 and d5, White on d4 and e5.
func (Game) Start() Pos {
	const d4, e4, d5, e5 = 27, 28, 35, 36
	return Pos{Mine: 1<<e4 | 1<<d5, Opp: 1<<d4 | 1<<e5}
}

// Key orders interned sample rows. It is not unique (two bitboards do not fit in 64 bits).
func (Game) Key(p Pos) uint64 { return p.Mine ^ p.Opp*0x9e3779b97f4a7c15 }

// flips is the mask of opponent discs a placement on cell would turn, or 0 if illegal.
func (Game) flips(p Pos, cell int) uint64 {
	if (p.Mine|p.Opp)>>cell&1 == 1 {
		return 0
	}
	r, c := cell/Size, cell%Size
	var all uint64
	for _, d := range dirs {
		var line uint64
		rr, cc := r+d[0], c+d[1]
		for rr >= 0 && rr < Size && cc >= 0 && cc < Size {
			i := rr*Size + cc
			if p.Opp>>i&1 == 1 {
				line |= 1 << i
				rr += d[0]
				cc += d[1]
				continue
			}
			if line != 0 && p.Mine>>i&1 == 1 {
				all |= line
			}
			break
		}
	}
	return all
}

func (g Game) hasPlacement(p Pos) bool {
	for cell := range N {
		if g.flips(p, cell) != 0 {
			return true
		}
	}
	return false
}

func (g Game) Legal(p Pos, dst []int) []int {
	for cell := range N {
		if g.flips(p, cell) != 0 {
			dst = append(dst, cell)
		}
	}
	if len(dst) == 0 {
		dst = append(dst, Pass)
	}
	return dst
}

// Outcome: the game ends only when neither side can place; more discs wins.
func (g Game) Outcome(p Pos) (int, bool) {
	if g.hasPlacement(p) {
		return 0, false
	}
	if g.hasPlacement(Pos{Mine: p.Opp, Opp: p.Mine}) {
		return 0, false
	}
	mine, opp := bits.OnesCount64(p.Mine), bits.OnesCount64(p.Opp)
	switch {
	case mine > opp:
		return 1, true
	case mine < opp:
		return -1, true
	default:
		return 0, true
	}
}

func (g Game) Play(p Pos, action int) Pos {
	mine, opp := p.Mine, p.Opp
	if action != Pass {
		flipped := g.flips(p, action)
		mine |= 1<<action | flipped
		opp &^= flipped
	}
	return Pos{Mine: opp, Opp: mine}
}

func (Game) Layout() game.Layout { return game.Layout{Channels: 2, Rows: Size, Cols: Size} }

func (Game) Observe(p Pos, obs []bool) {
	for i := range N {
		obs[i] = p.Mine>>i&1 == 1
		obs[N+i] = p.Opp>>i&1 == 1
	}
}

// ---- terminal UI: Black sits at the bottom, squares are named like a1 --------

func (Game) discs(p Pos, firstToMove bool) (black, white uint64) {
	if firstToMove {
		return p.Mine, p.Opp
	}
	return p.Opp, p.Mine
}

func square(cell int) string {
	return string(rune('a'+cell%Size)) + strconv.Itoa(cell/Size+1)
}

func (g Game) Render(p Pos, firstToMove bool) []string {
	black, white := g.discs(p, firstToMove)
	var rows []string
	for r := Size - 1; r >= 0; r-- {
		row := fmt.Sprintf("%2d |", r+1)
		for c := range Size {
			i := r*Size + c
			switch {
			case black>>i&1 == 1:
				row += " " + bold + red + "B" + reset
			case white>>i&1 == 1:
				row += " " + bold + cyan + "W" + reset
			default:
				row += " ."
			}
		}
		rows = append(rows, row)
	}
	files := "    "
	for c := range Size {
		files += " " + string(rune('a'+c))
	}
	return append(rows, files)
}

func (g Game) ParseAction(_ Pos, _ bool, text string) (int, bool) {
	text = strings.ToLower(text)
	if text == "pass" {
		return Pass, true
	}
	if len(text) != 2 || text[0] < 'a' || text[0] > 'h' || text[1] < '1' || text[1] > '8' {
		return 0, false
	}
	return int(text[1]-'1')*Size + int(text[0]-'a'), true
}

func (Game) ActionName(_ Pos, _ bool, action int) string {
	if action == Pass {
		return "pass"
	}
	return square(action)
}

func (Game) InputHint() string { return "e.g. c4, or pass" }

// ProbabilityBoard shows each legal cell's move probability; pass is a last line.
func (g Game) ProbabilityBoard(p Pos, probs []float64) []string {
	var rows []string
	for r := Size - 1; r >= 0; r-- {
		row := ""
		for c := range Size {
			i := r*Size + c
			if g.flips(p, i) == 0 {
				row += "     ."
			} else {
				row += fmt.Sprintf("  %3.0f%%", 100*probs[i])
			}
		}
		rows = append(rows, row)
	}
	if !g.hasPlacement(p) {
		rows = append(rows, fmt.Sprintf("  pass  %3.0f%%", 100*probs[Pass]))
	}
	return rows
}
