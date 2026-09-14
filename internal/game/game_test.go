package game

import (
	"math/rand/v2"
	"testing"
)

func TestPositionCount(t *testing.T) {
	// 5,478 reachable boards in total; 4,520 of them are non-terminal decision points.
	if n := len(Positions()); n != 4520 {
		t.Fatalf("got %d positions, want 4520", n)
	}
}

func TestEmptyBoardIsADraw(t *testing.T) {
	if v := Minimax(Pos{}); v != 0 {
		t.Fatalf("minimax(empty) = %d, want 0", v)
	}
}

func TestTerminal(t *testing.T) {
	if v, over := (Pos{Mine: 0b000011000, Opp: 0b000000111}).Terminal(); !over || v != -1 {
		t.Fatalf("opponent row: got (%d, %v)", v, over)
	}
	if _, over := (Pos{}).Terminal(); over {
		t.Fatal("empty board reported as over")
	}
}

func TestPlayFlipsPerspective(t *testing.T) {
	if got := (Pos{}).Play(4); got != (Pos{Mine: 0, Opp: 1 << 4}) {
		t.Fatalf("got %+v", got)
	}
}

func TestPerfectPlayerNeverLoses(t *testing.T) {
	rng := rand.New(rand.NewPCG(0, 0))
	for range 300 {
		if PlayGame(PerfectPlayer, RandomPlayer, rng) < 0 {
			t.Fatal("perfect X lost")
		}
		if PlayGame(RandomPlayer, PerfectPlayer, rng) > 0 {
			t.Fatal("perfect O lost")
		}
	}
}

func TestRandomVsRandomFavoursX(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	wins := 0
	const n = 4000
	for range n {
		if PlayGame(RandomPlayer, RandomPlayer, rng) == 1 {
			wins++
		}
	}
	if rate := float64(wins) / n; rate < 0.55 || rate > 0.62 {
		t.Fatalf("X win rate %.3f, want ~0.585", rate)
	}
}
