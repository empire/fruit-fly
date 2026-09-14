package hexapawn

import (
	"math/rand/v2"
	"testing"

	"github.com/empire/fruit-fly/internal/game"
)

func compile(t *testing.T, cols, rows int) *game.Tree {
	t.Helper()
	g, err := New(cols, rows)
	if err != nil {
		t.Fatal(err)
	}
	tree, err := game.Compile(g)
	if err != nil {
		t.Fatal(err)
	}
	return tree
}

// Counts and values were computed independently of this package. Gardner's 3x3 original is a
// win for the second player; 4x3 is a different game from 3x4, which catches rows/cols mix-ups.
func TestSizesAndValues(t *testing.T) {
	for _, tc := range []struct{ cols, rows, decisions, value int }{
		{3, 3, 70, -1},
		{3, 4, 479, -1},
		{4, 3, 486, 1},
	} {
		tree := compile(t, tc.cols, tc.rows)
		if tree.Decisions != tc.decisions || int(tree.Value[tree.Root]) != tc.value {
			t.Errorf("%dx%d: %d decisions, value %d; want %d, %d",
				tc.cols, tc.rows, tree.Decisions, tree.Value[tree.Root], tc.decisions, tc.value)
		}
	}
}

func TestNoDraws(t *testing.T) {
	tree := compile(t, 3, 4)
	rng := rand.New(rand.NewPCG(0, 0))
	for range 500 {
		if game.PlayGame(tree, game.RandomMove, game.RandomMove, rng) == 0 {
			t.Fatal("a game ended in a draw")
		}
	}
}

func TestPerfectSecondPlayerAlwaysWins(t *testing.T) {
	tree := compile(t, 3, 4)
	rng := rand.New(rand.NewPCG(1, 1))
	for range 300 {
		if game.PlayGame(tree, game.RandomMove, game.PerfectMove, rng) != -1 {
			t.Fatal("perfect second player did not win")
		}
	}
}

func TestOpeningMoves(t *testing.T) {
	g, _ := New(3, 4)
	p := g.Start()
	if got := g.Legal(p, nil); len(got) != 3 {
		t.Fatalf("opening moves %v, want the three forward moves", got)
	}
	// a1a2 moves the first pawn forward. Rotated to the second player's view, a2 (cell 3)
	// becomes cell 8 and the untouched b1, c1 (cells 1, 2) become cells 10, 9.
	next := g.Play(p, 0*3+Forward)
	if next.Opp != 1<<8|1<<9|1<<10 {
		t.Fatalf("after a1a2 the opponent mask is %b", next.Opp)
	}
}

func TestParseAndNameRoundTrip(t *testing.T) {
	g, _ := New(3, 4)
	tree := compile(t, 3, 4)
	rng := rand.New(rand.NewPCG(2, 2))
	for range 200 {
		first := true
		p := g.Start()
		for {
			if _, over := g.Outcome(p); over {
				break
			}
			legal := g.Legal(p, nil)
			a := legal[rng.IntN(len(legal))]
			name := g.ActionName(p, first, a)
			if back, ok := g.ParseAction(p, first, name); !ok || back != a {
				t.Fatalf("action %d named %q parsed back as %d (%v)", a, name, back, ok)
			}
			p = g.Play(p, a)
			first = !first
		}
	}
	if _, ok := tree.ParseAction(tree.Root, true, "a1b2"); ok {
		t.Fatal("a1b2 is not a legal opening move")
	}
}
