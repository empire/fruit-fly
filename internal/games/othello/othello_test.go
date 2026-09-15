package othello

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/empire/fruit-fly/internal/game"
)

// sq is algebraic notation from the first player's view (a1 = 0). Independently
// of Play's perspective flip, c4 is always cell 26.
func sq(name string) int { return int(name[1]-'1')*8 + int(name[0]-'a') }

// Opening squares are the four well-known first moves of WOF 8x8 Othello.
func TestOpeningMoves(t *testing.T) {
	got := New().Legal(New().Start(), nil)
	want := []int{sq("d3"), sq("c4"), sq("f5"), sq("e6")}
	if !slices.Equal(got, want) {
		t.Fatalf("opening moves %v, want %v (d3, c4, f5, e6)", got, want)
	}
}

func TestPlayFlipsDiscsAndPerspective(t *testing.T) {
	// Black plays c4 and flips d4. After the turn hand-off, White to move sees
	// its remaining disc on e5 as Mine.
	next := New().Play(New().Start(), sq("c4"))
	if next.Mine != 1<<sq("e5") {
		t.Fatalf("white's mask after c4 is %b, want e5", next.Mine)
	}
	wantOpp := uint64(1<<sq("c4") | 1<<sq("d4") | 1<<sq("e4") | 1<<sq("d5"))
	if next.Opp != wantOpp {
		t.Fatalf("black's mask after c4 is %b, want c4,d4,e4,d5", next.Opp)
	}
}

func TestWhiteRepliesAfterC4(t *testing.T) {
	// After the standard c4 opening, White has exactly c3, e3 and c5.
	next := New().Play(New().Start(), sq("c4"))
	got := New().Legal(next, nil)
	want := []int{sq("c3"), sq("e3"), sq("c5")}
	if !slices.Equal(got, want) {
		t.Fatalf("white replies %v, want %v (c3, e3, c5)", got, want)
	}
}

func TestMustPass(t *testing.T) {
	// Black occupies b2–b8; White sits on a1. Black has no sandwich; White can
	// play c3 (through b2 onto a1), so Black must pass.
	p := Pos{Mine: 1<<sq("b2") | 1<<sq("b3") | 1<<sq("b4") | 1<<sq("b5") | 1<<sq("b6") | 1<<sq("b7") | 1<<sq("b8"), Opp: 1 << sq("a1")}
	if v, over := New().Outcome(p); over || v != 0 {
		t.Fatalf("must-pass position reported as over (%d, %v)", v, over)
	}
	if got := New().Legal(p, nil); !slices.Equal(got, []int{Pass}) {
		t.Fatalf("legal %v, want only pass", got)
	}
	if got := New().Legal(New().Play(p, Pass), nil); !slices.Equal(got, []int{sq("c3")}) {
		t.Fatalf("after pass White's moves %v, want c3", got)
	}
}

func TestOutcomeCountsDiscs(t *testing.T) {
	// A full board: neither side can place, so the disc count decides.
	if v, over := New().Outcome(Pos{Mine: 0x00000000ffffffff, Opp: 0xffffffff00000000}); !over || v != 0 {
		t.Fatalf("32 vs 32: got (%d, %v), want a draw", v, over)
	}
	if v, over := New().Outcome(Pos{Mine: 1, Opp: ^uint64(1)}); !over || v != -1 {
		t.Fatalf("1 vs 63: got (%d, %v), want a loss", v, over)
	}
}

func TestSample(t *testing.T) {
	tree, err := game.Sample(New(), SampleSize, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !tree.Partial || tree.Decisions != SampleSize {
		t.Fatalf("sample: Partial=%v decisions=%d, want true / %d", tree.Partial, tree.Decisions, SampleSize)
	}
	w := tree.NewWalk()
	if w.Over || w.Row < 0 {
		t.Fatal("start should be an interned decision")
	}
	if got := New().Legal(New().Start(), nil); !slices.Equal(w.Legal, got) {
		t.Fatalf("walk legal %v, want %v", w.Legal, got)
	}
	// A full random game must finish using the rules, including off-tree positions.
	rng := rand.New(rand.NewPCG(3, 3))
	for !w.Over {
		w.Step(w.Legal[rng.IntN(len(w.Legal))])
	}
}

func TestParseAndNameRoundTrip(t *testing.T) {
	g := New()
	rng := rand.New(rand.NewPCG(2, 2))
	for range 50 {
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
	a, ok := g.ParseAction(g.Start(), true, "a1")
	if !ok || slices.Contains(g.Legal(g.Start(), nil), a) {
		t.Fatal("a1 must parse as a square but is not a legal opening move")
	}
}
