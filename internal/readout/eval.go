package readout

import (
	"math/rand/v2"

	"github.com/empire/fruit-fly/internal/game"
)

// Result counts outcomes from the player's point of view, as fractions of games.
type Result struct{ Win, Draw, Loss float64 }

// Score is a player's record against both reference opponents.
type Score struct{ VsRandom, VsPerfect Result }

// RandomChooser plays uniformly random legal moves: the baseline.
func RandomChooser() Chooser {
	return func(w *game.Walk, rng *rand.Rand) int {
		return w.Legal[rng.IntN(len(w.Legal))]
	}
}

// Evaluate plays n games on t against a random and a perfect player, half moving first.
// On a sample tree there is no perfect player, so VsPerfect is left zero.
func Evaluate(t *game.Tree, player Chooser, n int, seed uint64) Score {
	rng := rand.New(rand.NewPCG(seed, 2))
	var s Score
	opponents := []Opponent{Random, Perfect}
	if t.Partial {
		opponents = []Opponent{Random}
	}
	for _, o := range opponents {
		var res Result
		for i := range n {
			flyFirst := i%2 == 0
			outcomeFirst, _ := playGame(t, player, nil, o, flyFirst, rng, false)
			outcome := outcomeFirst
			if !flyFirst {
				outcome = -outcomeFirst
			}
			switch outcome {
			case 1:
				res.Win++
			case 0:
				res.Draw++
			default:
				res.Loss++
			}
		}
		res = Result{res.Win / float64(n), res.Draw / float64(n), res.Loss / float64(n)}
		if o == Random {
			s.VsRandom = res
		} else {
			s.VsPerfect = res
		}
	}
	return s
}
