// Package play is the terminal game against the fly, for any game.Tree.
package play

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/empire/fruit-fly/internal/connectome"
	"github.com/empire/fruit-fly/internal/features"
	"github.com/empire/fruit-fly/internal/game"
	"github.com/empire/fruit-fly/internal/readout"
	"github.com/empire/fruit-fly/internal/retina"
	"github.com/empire/fruit-fly/internal/sim"
)

const (
	bold  = "\033[1m"
	dim   = "\033[2m"
	reset = "\033[0m"
)

// Fly is the opponent: cached (or live) brain activity plus the trained readout.
type Fly struct {
	Readout *readout.Readout
	Cache   *features.Cache
	Meta    *connectome.Meta
	Brain   *sim.Brain // used on cache misses and when -live
	Eyes    *retina.Eyes
}

// Game runs one game of t. The human types moves on in; everything is written to out.
func Game(fly *Fly, t *game.Tree, humanFirst bool, in io.Reader, out io.Writer) {
	scanner := bufio.NewScanner(in)
	w := t.NewWalk()
	fmt.Fprintf(out, "\n  %s: you move %s, the fly moves %s\n\n", t.Name, order(humanFirst), order(!humanFirst))
	showBoard(out, w)
	for {
		if w.Over {
			switch {
			case w.Value == 0:
				fmt.Fprint(out, "\n  draw\n\n")
			case (w.Value < 0) == (w.First == humanFirst):
				fmt.Fprint(out, "\n  the fly wins\n\n")
			default:
				fmt.Fprint(out, "\n  you win\n\n")
			}
			return
		}
		var action int
		if w.First == humanFirst {
			var ok bool
			if action, ok = ask(scanner, out, w); !ok {
				return
			}
		} else {
			action = fly.move(out, w)
		}
		w.Step(action)
		fmt.Fprintln(out)
		showBoard(out, w)
	}
}

func order(first bool) string {
	if first {
		return "first"
	}
	return "second"
}

func showBoard(out io.Writer, w *game.Walk) {
	for _, row := range w.Render() {
		fmt.Fprintln(out, "   "+row)
	}
}

func ask(scanner *bufio.Scanner, out io.Writer, w *game.Walk) (int, bool) {
	for {
		fmt.Fprintf(out, "  your move (%s, q to quit): ", w.Tree.InputHint())
		if !scanner.Scan() {
			return 0, false
		}
		text := strings.TrimSpace(scanner.Text())
		if text == "q" {
			return 0, false
		}
		if action, ok := w.ParseAction(text); ok {
			return action, true
		}
		fmt.Fprintln(out, "  that move isn't legal")
	}
}

func (f *Fly) move(out io.Writer, w *game.Walk) int {
	var counts []uint8
	var regions [3][]int32

	if w.Row >= 0 && f.Cache != nil && w.Row < len(f.Cache.Counts) && f.Brain == nil {
		counts, regions = f.Cache.Counts[w.Row], f.Cache.Regions[w.Row]
	} else if w.Row >= 0 && f.Cache != nil && w.Row < len(f.Cache.Counts) && f.Brain != nil {
		fmt.Fprintf(out, "%s  simulating %d neurons for %d ticks ...", dim, f.Brain.G.N, f.Brain.P.Ticks)
		start := time.Now()
		run := f.Brain.Run(f.Eyes.Stimulus(w.Obs()))
		fmt.Fprintf(out, " %.1fs, matches cache: %v%s\n", time.Since(start).Seconds(),
			slices.Equal(run.ReadoutCounts, f.Cache.Counts[w.Row]), reset)
		counts, regions = run.ReadoutCounts, run.RegionSpikes
	} else {
		fmt.Fprintf(out, "%s  simulating %d neurons for %d ticks ...", dim, f.Brain.G.N, f.Brain.P.Ticks)
		start := time.Now()
		run := f.Brain.Run(f.Eyes.Stimulus(w.Obs()))
		fmt.Fprintf(out, " %.1fs%s\n", time.Since(start).Seconds(), reset)
		counts, regions = run.ReadoutCounts, run.RegionSpikes
	}

	var probs []float64
	var action int
	if w.Row < 0 {
		x := f.Readout.ScaleCounts(counts)
		probs = f.Readout.ProbsOn(x, w.Legal)
		action = greedyLegal(probs, w.Legal)
	} else {
		probs = f.Readout.ProbsWalk(w)
		action = f.Readout.GreedyWalk(w)
	}
	name := func(a int) string { return w.ActionName(a) }

	fmt.Fprintf(out, "\n  %sfly brain activity%s (spikes per tick)\n", bold, reset)
	for r, name := range connectome.Regions {
		fmt.Fprintf(out, "   %7s %s peak %d\n", name, sparkline(regions[r]), slices.Max(regions[r]))
	}

	fmt.Fprintf(out, "\n  %smove probabilities%s\n", bold, reset)
	if board, ok := w.ProbabilityBoard(probs); ok {
		for _, line := range board {
			fmt.Fprintln(out, "  "+line)
		}
	} else {
		legal := slices.Clone(w.Legal)
		slices.SortStableFunc(legal, func(a, b int) int { return -cmpFloat(probs[a], probs[b]) })
		for _, a := range legal[:min(8, len(legal))] {
			fmt.Fprintf(out, "   %-8s %3.0f%%\n", name(a), 100*probs[a])
		}
		if len(legal) > 8 {
			fmt.Fprintf(out, "   %s... %d more%s\n", dim, len(legal)-8, reset)
		}
	}

	x := f.Readout.X[0]
	if w.Row >= 0 {
		x = f.Readout.X[w.Row]
	} else {
		x = f.Readout.ScaleCounts(counts)
	}
	type voter struct {
		neuron int
		vote   float64
	}
	var voters []voter
	for k, n := range counts {
		if n > 0 {
			voters = append(voters, voter{k, f.Readout.W[action][k] * x[k]})
		}
	}
	slices.SortFunc(voters, func(a, b voter) int { return -cmpFloat(a.vote, b.vote) })
	fmt.Fprintf(out, "\n  %sfiring neurons voting for %s%s (%d of %d readout neurons fired)\n",
		bold, name(action), reset, len(voters), len(counts))
	if len(voters) == 0 {
		fmt.Fprintf(out, "   %snone fired: the choice comes from which neurons stayed quiet%s\n", dim, reset)
	}
	for _, v := range voters[:min(6, len(voters))] {
		kind := "motor"
		if f.Meta.ReadoutClasses[v.neuron] == "descending_neuron" {
			kind = "descending"
		}
		neuron := f.Meta.ReadoutTypes[v.neuron]
		fmt.Fprintf(out, "   %22s  %-10s %3d spikes  %svote %+.2f%s\n",
			neuron[:min(22, len(neuron))], kind, counts[v.neuron], dim, v.vote, reset)
	}
	fmt.Fprintf(out, "\n  %sthe fly plays %s%s\n\n", bold, name(action), reset)
	return action
}

func greedyLegal(p []float64, legal []int) int {
	best := legal[0]
	for _, c := range legal[1:] {
		if p[c] > p[best] {
			best = c
		}
	}
	return best
}

func cmpFloat(a, b float64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

func sparkline(values []int32) string {
	const bars = " ▁▂▃▄▅▆▇█"
	levels := []rune(bars)
	top := max(slices.Max(values), 1)
	var b strings.Builder
	for _, v := range values {
		b.WriteRune(levels[int(math.Round(float64(v)/float64(top)*float64(len(levels)-1)))])
	}
	return b.String()
}
