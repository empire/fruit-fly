// Command fly plays tic-tac-toe with a simulated MaleCNS v1.0 fruit-fly brain.
//
// Run the pipeline in order: download, build, features, train, eval, play.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/empire/fruit-fly/internal/analyze"
	"github.com/empire/fruit-fly/internal/connectome"
	"github.com/empire/fruit-fly/internal/download"
	"github.com/empire/fruit-fly/internal/features"
	"github.com/empire/fruit-fly/internal/play"
	"github.com/empire/fruit-fly/internal/readout"
	"github.com/empire/fruit-fly/internal/sim"
)

const (
	rawDir       = "data/raw"
	processedDir = "data/processed"
	featuresPath = "data/features.bin"
)

func readoutPath(name string) string { return "data/readout_" + name + ".bin" }

// loadReadout builds a readout on a feature set and loads its trained weights.
func loadReadout(name string) (*readout.Readout, error) {
	x, err := readout.FeatureMatrix(name, featuresPath)
	if err != nil {
		return nil, err
	}
	r := readout.New(x)
	return r, r.Load(readoutPath(name))
}

func loadBrain() (*sim.Brain, *connectome.Meta, error) {
	g, meta, err := connectome.Load(processedDir)
	if err != nil {
		return nil, nil, err
	}
	return sim.New(g, sim.DefaultParams()), meta, nil
}

type command struct {
	help string
	run  func(args []string) error
}

var commands = map[string]command{}
var order []string

func register(name, help string, run func(args []string) error) {
	commands[name] = command{help, run}
	order = append(order, name)
}

func init() {
	register("download", "fetch the MaleCNS v1.0 tables (~1.1 GB)", func([]string) error {
		return download.All(rawDir)
	})
	register("build", "feathers -> signed sparse graph", func([]string) error {
		_, _, err := connectome.Build(rawDir, processedDir)
		return err
	})
	register("bench", "time the simulation on this machine", func([]string) error {
		brain, _, err := loadBrain()
		if err != nil {
			return err
		}
		features.Bench(brain)
		return nil
	})
	register("features", "simulate every position once and cache readout spikes", func([]string) error {
		brain, _, err := loadBrain()
		if err != nil {
			return err
		}
		c, err := features.Compute(brain, featuresPath)
		if err != nil {
			return err
		}
		features.Sanity(c)
		return nil
	})
	register("sanity", "does the brain's output depend on the board?", func([]string) error {
		c, err := features.Load(featuresPath)
		if err != nil {
			return err
		}
		features.Sanity(c)
		return nil
	})
	register("train", "train the linear readout with self-play REINFORCE", func(args []string) error {
		fs := flags("train")
		set := fs.String("features", "all", "brain, random, board or all")
		games := fs.Int("games", 300_000, "training games per readout")
		seed := fs.Uint64("seed", 0, "random seed for training games")
		fs.Parse(args)
		names := readout.FeatureSets
		if *set != "all" {
			names = []string{*set}
		}
		for _, name := range names {
			x, err := readout.FeatureMatrix(name, featuresPath)
			if err != nil {
				return err
			}
			cfg := readout.DefaultTrainConfig()
			cfg.Games = *games
			cfg.Seed = *seed
			if err := readout.Train(name, x, cfg).Save(readoutPath(name)); err != nil {
				return err
			}
		}
		return nil
	})
	register("eval", "compare the fly with baselines", func(args []string) error {
		fs := flags("eval")
		games := fs.Int("games", 4000, "games per column")
		fs.Parse(args)
		row := func(label string, s readout.Score) {
			r, p := s.VsRandom, s.VsPerfect
			fmt.Printf("%-34s %5.1f%% %5.1f%% %5.1f%%   %5.1f%% %5.1f%% %5.1f%%\n", label,
				100*r.Win, 100*r.Draw, 100*r.Loss, 100*p.Win, 100*p.Draw, 100*p.Loss)
		}
		fmt.Printf("%-34s %-22s   %s\n", "", "   vs random player", "   vs perfect player")
		fmt.Printf("%-34s %6s %6s %6s   %6s %6s %6s\n", "player", "win", "draw", "loss", "win", "draw", "loss")
		row("random moves (baseline)", readout.Evaluate(readout.RandomChooser, *games, 0))
		for _, name := range readout.FeatureSets {
			r, err := loadReadout(name)
			if err != nil {
				fmt.Printf("%-34s (not trained: %v)\n", readout.Labels[name], err)
				continue
			}
			row(readout.Labels[name], readout.Evaluate(r.Chooser(), *games, 0))
		}
		fmt.Printf("\n%d games per column, half as X and half as O. The perfect player never loses,\n"+
			"so 'loss' there is how often the player blunders.\n", *games)
		return nil
	})
	register("analyze", "why does the brain readout play worse? three experiments", func([]string) error {
		sets := map[string][][]float64{}
		for _, name := range readout.FeatureSets {
			x, err := readout.FeatureMatrix(name, featuresPath)
			if err != nil {
				return err
			}
			sets[name] = x
		}
		analyze.Run(sets)
		return nil
	})
	register("play", "play against the fly in the terminal", func(args []string) error {
		fs := flags("play")
		sideFlag := fs.String("side", "x", "your side: x (moves first) or o")
		live := fs.Bool("live", false, "re-run the full simulation on every fly move")
		fs.Parse(args)
		r, err := loadReadout("brain")
		if err != nil {
			return err
		}
		cache, err := features.Load(featuresPath)
		if err != nil {
			return err
		}
		brain, meta, err := loadBrain()
		if err != nil {
			return err
		}
		fly := &play.Fly{Readout: r, Cache: cache, Meta: meta}
		if *live {
			fly.Brain = brain
		}
		play.Game(fly, *sideFlag != "o", os.Stdin, os.Stdout)
		return nil
	})
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: fly <command> [flags]\n\ncommands:")
	for _, name := range order {
		fmt.Fprintf(os.Stderr, "  %-9s %s\n", name, commands[name].help)
	}
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd, ok := commands[os.Args[1]]
	if !ok {
		usage()
		os.Exit(2)
	}
	if err := cmd.run(os.Args[2:]); err != nil {
		fmt.Fprintln(os.Stderr, "fly:", err)
		os.Exit(1)
	}
}

// flags creates a flag set for a subcommand that exits on -h.
func flags(name string) *flag.FlagSet {
	return flag.NewFlagSet("fly "+name, flag.ExitOnError)
}
