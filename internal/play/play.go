// Package play is the terminal game against the fly.
package play

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/empire/fruit-fly/internal/connectome"
	"github.com/empire/fruit-fly/internal/features"
	"github.com/empire/fruit-fly/internal/game"
	"github.com/empire/fruit-fly/internal/readout"
	"github.com/empire/fruit-fly/internal/sim"
)

const (
	bold  = "\033[1m"
	dim   = "\033[2m"
	red   = "\033[31m"
	cyan  = "\033[36m"
	reset = "\033[0m"
)

// Fly is the opponent: cached (or live) brain activity plus the trained readout.
type Fly struct {
	Readout *readout.Readout
	Cache   *features.Cache
	Meta    *connectome.Meta
	Brain   *sim.Brain // nil unless playing live
}

// Game runs one game. The human types cells 1-9 on in; everything is written to out.
func Game(fly *Fly, humanIsX bool, in io.Reader, out io.Writer) {
	scanner := bufio.NewScanner(in)
	var pos game.Pos
	xToMove := true
	fmt.Fprintf(out, "\n  you are %s, the fly is %s\n\n", side(humanIsX), side(!humanIsX))
	showBoard(out, pos, xToMove)
	for {
		result, over := pos.Terminal()
		if over {
			switch {
			case result == 0:
				fmt.Fprint(out, "\n  draw\n\n")
			case xToMove == humanIsX: // the side to move has just lost
				fmt.Fprint(out, "\n  the fly wins\n\n")
			default:
				fmt.Fprint(out, "\n  you win\n\n")
			}
			return
		}
		var cell int
		if xToMove == humanIsX {
			var ok bool
			if cell, ok = ask(scanner, out, pos); !ok {
				return
			}
		} else {
			cell = fly.move(out, pos)
		}
		pos = pos.Play(cell)
		xToMove = !xToMove
		fmt.Fprintln(out)
		showBoard(out, pos, xToMove)
	}
}

func side(x bool) string {
	if x {
		return "X"
	}
	return "O"
}

func showBoard(out io.Writer, pos game.Pos, xToMove bool) {
	for i, row := range game.Render(pos, xToMove) {
		row = strings.ReplaceAll(row, "X", bold+red+"X"+reset)
		row = strings.ReplaceAll(row, "O", bold+cyan+"O"+reset)
		fmt.Fprintln(out, "   "+row)
		if i < 2 {
			fmt.Fprintln(out, "   ---+---+---")
		}
	}
}

func ask(scanner *bufio.Scanner, out io.Writer, pos game.Pos) (int, bool) {
	for {
		fmt.Fprint(out, "  your move (1-9, q to quit): ")
		if !scanner.Scan() {
			return 0, false
		}
		text := strings.TrimSpace(scanner.Text())
		if text == "q" {
			return 0, false
		}
		if n, err := strconv.Atoi(text); err == nil && slices.Contains(pos.Legal(), n-1) {
			return n - 1, true
		}
		fmt.Fprintln(out, "  that cell isn't free")
	}
}

func (f *Fly) move(out io.Writer, pos game.Pos) int {
	row := readout.GetTables().RowOf(pos)
	counts, regions := f.Cache.Counts[row], f.Cache.Regions[row]

	if f.Brain != nil {
		fmt.Fprintf(out, "%s  simulating %d neurons for %d ticks ...", dim, f.Brain.G.N, f.Brain.P.Ticks)
		start := time.Now()
		live := f.Brain.Run(pos)
		fmt.Fprintf(out, " %.1fs, matches cache: %v%s\n", time.Since(start).Seconds(),
			slices.Equal(live.ReadoutCounts, counts), reset)
		counts, regions = live.ReadoutCounts, live.RegionSpikes
	}

	probs := f.Readout.Probs(row)
	cell := f.Readout.Greedy(row)

	fmt.Fprintf(out, "\n  %sfly brain activity%s (spikes per tick)\n", bold, reset)
	for r, name := range connectome.Regions {
		fmt.Fprintf(out, "   %7s %s peak %d\n", name, sparkline(regions[r]), slices.Max(regions[r]))
	}

	fmt.Fprintf(out, "\n  %smove probabilities%s\n", bold, reset)
	for r := range 3 {
		fmt.Fprint(out, "  ")
		for c := r * 3; c < r*3+3; c++ {
			if !readout.GetTables().Legal[row][c] {
				fmt.Fprint(out, "     .")
			} else {
				fmt.Fprintf(out, "  %3.0f%%", 100*probs[c])
			}
		}
		fmt.Fprintln(out)
	}

	// Which firing readout neurons pushed the chosen cell up? weight × standardized activity.
	// (A silent neuron can "vote" too, by being quieter than usual; we only list ones that fired.)
	x := f.Readout.X[row]
	type voter struct {
		neuron int
		vote   float64
	}
	var voters []voter
	for k, n := range counts {
		if n > 0 {
			voters = append(voters, voter{k, f.Readout.W[cell][k] * x[k]})
		}
	}
	slices.SortFunc(voters, func(a, b voter) int { return -cmpFloat(a.vote, b.vote) })
	fmt.Fprintf(out, "\n  %sfiring neurons voting for cell %d%s (%d of %d readout neurons fired)\n",
		bold, cell+1, reset, len(voters), len(counts))
	if len(voters) == 0 {
		fmt.Fprintf(out, "   %snone fired: the choice comes from which neurons stayed quiet%s\n", dim, reset)
	}
	for _, v := range voters[:min(6, len(voters))] {
		kind := "motor"
		if f.Meta.ReadoutClasses[v.neuron] == "descending_neuron" {
			kind = "descending"
		}
		name := f.Meta.ReadoutTypes[v.neuron]
		fmt.Fprintf(out, "   %22s  %-10s %3d spikes  %svote %+.2f%s\n",
			name[:min(22, len(name))], kind, counts[v.neuron], dim, v.vote, reset)
	}
	fmt.Fprintf(out, "\n  %sthe fly plays %d%s\n\n", bold, cell+1, reset)
	return cell
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
