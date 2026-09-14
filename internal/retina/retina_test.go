package retina

import (
	"slices"
	"testing"

	"github.com/empire/fruit-fly/internal/connectome"
	"github.com/empire/fruit-fly/internal/game"
)

func pool(from, n int32) []int32 {
	var p []int32
	for i := range n {
		p = append(p, from+i)
	}
	return p
}

func TestDealIsEvenAndDisjoint(t *testing.T) {
	g := &connectome.Graph{SensorPools: [][]int32{pool(0, 50), pool(100, 25)}}
	e, err := New(g, game.Layout{Channels: 2, Rows: 2, Cols: 3}, DefaultDrive)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[int32]bool{}
	for ch, want := range []int{8, 4} { // 50/6 and 25/6, remainders dropped
		for cell, group := range e.Groups[ch] {
			if len(group) != want {
				t.Errorf("channel %d cell %d: %d neurons, want %d", ch, cell, len(group), want)
			}
			for _, n := range group {
				if seen[n] {
					t.Fatalf("neuron %d dealt twice", n)
				}
				seen[n] = true
			}
		}
	}
}

func TestStimulusLightsOnlyLitCells(t *testing.T) {
	g := &connectome.Graph{SensorPools: [][]int32{pool(0, 9), pool(9, 9)}}
	e, err := New(g, game.Layout{Channels: 2, Rows: 3, Cols: 3}, []float32{0.5, 0.7})
	if err != nil {
		t.Fatal(err)
	}
	obs := make([]bool, 18)
	obs[4] = true   // my piece on cell 4
	obs[9+2] = true // opponent piece on cell 2
	in := e.Stimulus(obs)
	if len(in) != 2 || !slices.Equal(in[0].Neurons, []int32{2 + 9}) || in[0].Current != 0.7 ||
		!slices.Equal(in[1].Neurons, []int32{4}) || in[1].Current != 0.5 {
		t.Fatalf("got %+v", in)
	}
}

func TestTooManyChannels(t *testing.T) {
	g := &connectome.Graph{SensorPools: [][]int32{pool(0, 9)}}
	if _, err := New(g, game.Layout{Channels: 2, Rows: 3, Cols: 3}, DefaultDrive); err == nil {
		t.Fatal("expected an error for 2 channels on 1 pool")
	}
}
