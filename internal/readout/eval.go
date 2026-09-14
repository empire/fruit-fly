package readout

import "math/rand/v2"

// Result counts outcomes from the player's point of view, as fractions of games.
type Result struct{ Win, Draw, Loss float64 }

// Score is a player's record against both reference opponents.
type Score struct{ VsRandom, VsPerfect Result }

// RandomChooser plays uniformly random legal moves: the baseline.
func RandomChooser(row int, rng *rand.Rand) int { return opponentMove(Random, row, rng) }

// Evaluate plays n games against a random and a perfect player, half as X and half as O.
func Evaluate(player Chooser, n int, seed uint64) Score {
	rng := rand.New(rand.NewPCG(seed, 2))
	var s Score
	for _, o := range []Opponent{Random, Perfect} {
		var res Result
		for i := range n {
			flyIsX := i%2 == 0
			outcomeX, _ := playGame(player, nil, o, flyIsX, rng, false)
			outcome := outcomeX
			if !flyIsX {
				outcome = -outcomeX
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
