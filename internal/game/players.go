package game

import "math/rand/v2"

// Player picks an action for a decision row.
type Player func(t *Tree, row int, rng *rand.Rand) int

// RandomMove picks uniformly among legal moves of an interned row.
func RandomMove(t *Tree, row int, rng *rand.Rand) int {
	legal := t.LegalActions(row)
	return legal[rng.IntN(len(legal))]
}

// PerfectMove picks uniformly among optimal moves. On a sample tree it falls back to random.
func PerfectMove(t *Tree, row int, rng *rand.Rand) int {
	if t.Partial || row < 0 || row >= len(t.Best) || t.Best[row] == nil {
		return RandomMove(t, row, rng)
	}
	var actions []int
	for a, ok := range t.Best[row] {
		if ok {
			actions = append(actions, a)
		}
	}
	if len(actions) == 0 {
		return RandomMove(t, row, rng)
	}
	return actions[rng.IntN(len(actions))]
}

// PlayGame plays one game and returns the result for the first player (+1, 0, -1).
func PlayGame(t *Tree, first, second Player, rng *rand.Rand) int {
	players := [2]Player{first, second}
	w := t.NewWalk()
	for turn := 0; ; turn++ {
		if w.Over {
			if turn%2 == 0 {
				return w.Value
			}
			return -w.Value
		}
		var action int
		if w.Row >= 0 {
			action = players[turn%2](t, w.Row, rng)
		} else {
			action = w.Legal[rng.IntN(len(w.Legal))]
		}
		w.Step(action)
	}
}
