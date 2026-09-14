// Package game implements tic-tac-toe rules, position enumeration and reference players.
//
// A position is always seen from the side to move: Mine and Opp are 9-bit masks
// (bit i = cell i, cells numbered 0..8 left-to-right, top-to-bottom). This is the same
// view the fly gets, so X-to-move and O-to-move positions share one encoding. They never
// collide, because the piece counts differ.
package game

import (
	"math/rand/v2"
	"slices"
	"strconv"
	"sync"
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

// Key packs a position into 18 bits: mine | opp<<9.
func (p Pos) Key() int { return int(p.Mine) | int(p.Opp)<<9 }

// Won reports whether a 9-bit mask contains a complete line.
func Won(mask uint16) bool {
	for _, l := range Lines {
		if mask&l == l {
			return true
		}
	}
	return false
}

// Legal returns the empty cells in increasing order.
func (p Pos) Legal() []int {
	var moves []int
	taken := p.Mine | p.Opp
	for c := range 9 {
		if taken>>c&1 == 0 {
			moves = append(moves, c)
		}
	}
	return moves
}

// Terminal returns the result for the side to move (-1 loss, 0 draw) and whether the game is over.
// The side to move can never have just won, so a win is impossible here.
func (p Pos) Terminal() (int, bool) {
	if Won(p.Opp) {
		return -1, true
	}
	if p.Mine|p.Opp == Full {
		return 0, true
	}
	return 0, false
}

// Play places a piece for the side to move and hands the turn over (the perspective flips).
func (p Pos) Play(cell int) Pos {
	return Pos{Mine: p.Opp, Opp: p.Mine | 1<<cell}
}

var positions = sync.OnceValue(func() []Pos {
	seen := map[Pos]bool{}
	stack := []Pos{{}}
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if _, over := p.Terminal(); over || seen[p] {
			continue
		}
		seen[p] = true
		for _, c := range p.Legal() {
			stack = append(stack, p.Play(c))
		}
	}
	all := make([]Pos, 0, len(seen))
	for p := range seen {
		all = append(all, p)
	}
	slices.SortFunc(all, func(a, b Pos) int { return a.Key() - b.Key() })
	return all
})

// Positions returns every reachable non-terminal position (4,520), sorted by Key.
// These are all the decisions a player can ever face.
func Positions() []Pos { return positions() }

var (
	minimaxMu    sync.Mutex
	minimaxCache = map[Pos]int{}
)

// Minimax returns the game value for the side to move under perfect play (+1, 0, -1).
func Minimax(p Pos) int {
	minimaxMu.Lock()
	defer minimaxMu.Unlock()
	return minimax(p)
}

func minimax(p Pos) int {
	if v, over := p.Terminal(); over {
		return v
	}
	if v, ok := minimaxCache[p]; ok {
		return v
	}
	best := -2
	for _, c := range p.Legal() {
		best = max(best, -minimax(p.Play(c)))
	}
	minimaxCache[p] = best
	return best
}

// BestMoves returns every move that achieves the minimax value.
func BestMoves(p Pos) []int {
	moves := p.Legal()
	scores := make([]int, len(moves))
	for i, c := range moves {
		scores[i] = -Minimax(p.Play(c))
	}
	top := slices.Max(scores)
	var best []int
	for i, c := range moves {
		if scores[i] == top {
			best = append(best, c)
		}
	}
	return best
}

// Player picks a cell for the side to move.
type Player func(p Pos, rng *rand.Rand) int

// RandomPlayer picks uniformly among legal moves.
func RandomPlayer(p Pos, rng *rand.Rand) int {
	moves := p.Legal()
	return moves[rng.IntN(len(moves))]
}

// PerfectPlayer picks uniformly among optimal moves. It never loses.
func PerfectPlayer(p Pos, rng *rand.Rand) int {
	moves := BestMoves(p)
	return moves[rng.IntN(len(moves))]
}

// PlayGame plays one game and returns +1 if X wins, -1 if O wins, 0 for a draw.
func PlayGame(x, o Player, rng *rand.Rand) int {
	var p Pos
	players := [2]Player{x, o}
	turn := 0
	for {
		if v, over := p.Terminal(); over {
			// v is for the side to move, which just lost or drew.
			if turn%2 == 0 {
				return v
			}
			return -v
		}
		p = p.Play(players[turn%2](p, rng))
		turn++
	}
}

// Render returns three text rows; empty cells show their 1-9 number.
func Render(p Pos, xToMove bool) [3]string {
	xs, os := p.Mine, p.Opp
	if !xToMove {
		xs, os = os, xs
	}
	var rows [3]string
	for r := range 3 {
		row := ""
		for c := r * 3; c < r*3+3; c++ {
			cell := strconv.Itoa(c + 1)
			switch {
			case xs>>c&1 == 1:
				cell = "X"
			case os>>c&1 == 1:
				cell = "O"
			}
			if c > r*3 {
				row += " | "
			}
			row += cell
		}
		rows[r] = " " + row
	}
	return rows
}
