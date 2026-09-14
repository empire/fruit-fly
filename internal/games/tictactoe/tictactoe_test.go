package tictactoe

import (
	"math/rand/v2"
	"testing"

	"github.com/empire/fruit-fly/internal/game"
)

func compile(t *testing.T) *game.Tree {
	t.Helper()
	tree, err := game.Compile(New())
	if err != nil {
		t.Fatal(err)
	}
	return tree
}

func TestPositionCount(t *testing.T) {
	// 5,478 reachable boards in total; 4,520 of them are non-terminal decision points.
	tree := compile(t)
	if tree.Decisions != 4520 || tree.Nodes() != 5478 {
		t.Fatalf("got %d decisions / %d nodes, want 4520 / 5478", tree.Decisions, tree.Nodes())
	}
}

func TestEmptyBoardIsADraw(t *testing.T) {
	tree := compile(t)
	if v := tree.Value[tree.Root]; v != 0 {
		t.Fatalf("minimax(empty) = %d, want 0", v)
	}
}

func TestOutcome(t *testing.T) {
	if v, over := New().Outcome(Pos{Mine: 0b000011000, Opp: 0b000000111}); !over || v != -1 {
		t.Fatalf("opponent row: got (%d, %v)", v, over)
	}
	if _, over := New().Outcome(Pos{}); over {
		t.Fatal("empty board reported as over")
	}
}

func TestPlayFlipsPerspective(t *testing.T) {
	if got := New().Play(Pos{}, 4); got != (Pos{Mine: 0, Opp: 1 << 4}) {
		t.Fatalf("got %+v", got)
	}
}

func TestObservationIsTheBoardBits(t *testing.T) {
	tree := compile(t)
	for row, obs := range tree.Obs {
		back, ok := tree.RowOfObservation(obs)
		if !ok || back != row {
			t.Fatalf("row %d: observation maps back to %d (%v)", row, back, ok)
		}
	}
}

func TestPerfectPlayerNeverLoses(t *testing.T) {
	tree := compile(t)
	rng := rand.New(rand.NewPCG(0, 0))
	for range 300 {
		if game.PlayGame(tree, game.PerfectMove, game.RandomMove, rng) < 0 {
			t.Fatal("perfect X lost")
		}
		if game.PlayGame(tree, game.RandomMove, game.PerfectMove, rng) > 0 {
			t.Fatal("perfect O lost")
		}
	}
}

func TestRandomVsRandomFavoursX(t *testing.T) {
	tree := compile(t)
	rng := rand.New(rand.NewPCG(1, 1))
	wins := 0
	const n = 4000
	for range n {
		if game.PlayGame(tree, game.RandomMove, game.RandomMove, rng) == 1 {
			wins++
		}
	}
	if rate := float64(wins) / n; rate < 0.55 || rate > 0.62 {
		t.Fatalf("X win rate %.3f, want ~0.585", rate)
	}
}
