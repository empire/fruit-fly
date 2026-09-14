package game

import "math/rand/v2"

// Player picks an action for a decision row.
type Player func(t *Tree, row int, rng *rand.Rand) int

// RandomMove picks uniformly among legal moves.
func RandomMove(t *Tree, row int, rng *rand.Rand) int { return pick(t.Next[row], nil, rng) }

// PerfectMove picks uniformly among optimal moves. It never does worse than the game's value.
func PerfectMove(t *Tree, row int, rng *rand.Rand) int { return pick(t.Next[row], t.Best[row], rng) }

func pick(next []int32, allowed []bool, rng *rand.Rand) int {
	var actions []int
	for a, n := range next {
		if n >= 0 && (allowed == nil || allowed[a]) {
			actions = append(actions, a)
		}
	}
	return actions[rng.IntN(len(actions))]
}

// PlayGame plays one game and returns the result for the first player (+1, 0, -1).
func PlayGame(t *Tree, first, second Player, rng *rand.Rand) int {
	players := [2]Player{first, second}
	node := t.Root
	for turn := 0; ; turn++ {
		if t.IsTerminal(node) {
			// Value is for the side to move, which is the first player on even turns.
			if turn%2 == 0 {
				return int(t.Value[node])
			}
			return -int(t.Value[node])
		}
		node = int(t.Next[node][players[turn%2](t, node, rng)])
	}
}
