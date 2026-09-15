// Command fly plays board games with a simulated MaleCNS v1.0 fruit-fly brain.
//
// Run the pipeline in order: download, build, then per game: features, train, eval, play.
// Pick the game with -game (see `fly games`); the default is tic-tac-toe.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/empire/fruit-fly/internal/analyze"
	"github.com/empire/fruit-fly/internal/connectome"
	"github.com/empire/fruit-fly/internal/download"
	"github.com/empire/fruit-fly/internal/features"
	"github.com/empire/fruit-fly/internal/featureset"
	"github.com/empire/fruit-fly/internal/game"
	"github.com/empire/fruit-fly/internal/games"
	"github.com/empire/fruit-fly/internal/play"
	"github.com/empire/fruit-fly/internal/readout"
	"github.com/empire/fruit-fly/internal/retina"
	"github.com/empire/fruit-fly/internal/sim"
)

const (
	dataDir      = "data"
	rawDir       = "data/raw"
	processedDir = "data/processed"
)

func featuresPath(t *game.Tree) string {
	return filepath.Join(games.DataDir(dataDir, t), "features.bin")
}

func readoutPath(t *game.Tree, set string) string {
	return filepath.Join(games.DataDir(dataDir, t), "readout_"+set+".bin")
}

// loadReadout builds a readout on a feature set and loads its trained weights.
func loadReadout(t *game.Tree, set string) (*readout.Readout, error) {
	x, err := featureset.Matrix(set, t, featuresPath(t))
	if err != nil {
		return nil, err
	}
	r := readout.New(t, x)
	return r, r.Load(readoutPath(t, set))
}

// loadBrain loads the connectome and fits the eyes to the game's board.
func loadBrain(t *game.Tree) (*sim.Brain, *retina.Eyes, *connectome.Meta, error) {
	g, meta, err := connectome.Load(processedDir)
	if err != nil {
		return nil, nil, nil, err
	}
	eyes, err := retina.New(g, t.Layout, retina.DefaultDrive)
	if err != nil {
		return nil, nil, nil, err
	}
	return sim.New(g, sim.DefaultParams()), eyes, meta, nil
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

// gameFlags creates a flag set for a subcommand with the -game flag. Call tree after Parse.
func gameFlags(name string) (fs *flag.FlagSet, tree func() (*game.Tree, error)) {
	fs = flags(name)
	spec := fs.String("game", "tictactoe", "game to play (see `fly games`)")
	return fs, func() (*game.Tree, error) { return games.Lookup(*spec) }
}

func init() {
	register("download", "fetch the MaleCNS v1.0 tables (~1.1 GB)", func([]string) error {
		return download.All(rawDir)
	})
	register("build", "feathers -> signed sparse graph", func([]string) error {
		_, _, err := connectome.Build(rawDir, processedDir)
		return err
	})
	register("games", "list the games", func([]string) error {
		for _, line := range games.Names() {
			fmt.Println("  " + line)
		}
		return nil
	})
	register("bench", "time the simulation on this machine", func(args []string) error {
		fs, tree := gameFlags("bench")
		fs.Parse(args)
		t, err := tree()
		if err != nil {
			return err
		}
		brain, eyes, _, err := loadBrain(t)
		if err != nil {
			return err
		}
		features.Bench(brain, eyes, t)
		return nil
	})
	register("features", "simulate every position once and cache readout spikes", func(args []string) error {
		fs, tree := gameFlags("features")
		fs.Parse(args)
		t, err := tree()
		if err != nil {
			return err
		}
		brain, eyes, _, err := loadBrain(t)
		if err != nil {
			return err
		}
		if t.Partial {
			fmt.Printf("%s: %d sampled positions\n", t.Name, t.Decisions)
		} else {
			fmt.Printf("%s: %d positions\n", t.Name, t.Decisions)
		}
		c, err := features.Compute(brain, eyes, t, featuresPath(t))
		if err != nil {
			return err
		}
		features.Sanity(c)
		return nil
	})
	register("sanity", "does the brain's output depend on the board?", func(args []string) error {
		fs, tree := gameFlags("sanity")
		fs.Parse(args)
		t, err := tree()
		if err != nil {
			return err
		}
		c, err := features.Load(featuresPath(t), t)
		if err != nil {
			return err
		}
		features.Sanity(c)
		return nil
	})
	register("train", "train the linear readout with self-play REINFORCE", func(args []string) error {
		fs, tree := gameFlags("train")
		set := fs.String("features", "all", "brain, random, board or all")
		gamesN := fs.Int("games", 300_000, "training games per readout")
		seed := fs.Uint64("seed", 0, "random seed for training games")
		fs.Parse(args)
		t, err := tree()
		if err != nil {
			return err
		}
		names := featureset.Names
		if *set != "all" {
			names = []string{*set}
		}
		for _, name := range names {
			x, err := featureset.Matrix(name, t, featuresPath(t))
			if err != nil {
				return err
			}
			cfg := readout.DefaultTrainConfig()
			cfg.Games = *gamesN
			cfg.Seed = *seed
			if err := readout.Train(name, t, x, cfg).Save(readoutPath(t, name)); err != nil {
				return err
			}
		}
		return nil
	})
	register("eval", "compare the fly with baselines", func(args []string) error {
		fs, tree := gameFlags("eval")
		gamesN := fs.Int("games", 4000, "games per column")
		fs.Parse(args)
		t, err := tree()
		if err != nil {
			return err
		}
		row := func(label string, s readout.Score) {
			r, p := s.VsRandom, s.VsPerfect
			if t.Partial {
				fmt.Printf("%-34s %5.1f%% %5.1f%% %5.1f%%      n/a    n/a    n/a\n", label,
					100*r.Win, 100*r.Draw, 100*r.Loss)
				return
			}
			fmt.Printf("%-34s %5.1f%% %5.1f%% %5.1f%%   %5.1f%% %5.1f%% %5.1f%%\n", label,
				100*r.Win, 100*r.Draw, 100*r.Loss, 100*p.Win, 100*p.Draw, 100*p.Loss)
		}
		fmt.Printf("%-34s %-22s   %s\n", t.Name, "   vs random player", "   vs perfect player")
		fmt.Printf("%-34s %6s %6s %6s   %6s %6s %6s\n", "player", "win", "draw", "loss", "win", "draw", "loss")
		row("random moves (baseline)", readout.Evaluate(t, readout.RandomChooser(), *gamesN, 0))
		for _, name := range featureset.Names {
			r, err := loadReadout(t, name)
			if err != nil {
				fmt.Printf("%-34s (not trained: %v)\n", featureset.Label(name, t), err)
				continue
			}
			row(featureset.Label(name, t), readout.Evaluate(t, r.Chooser(), *gamesN, 0))
		}
		if t.Partial {
			fmt.Printf("\n%d games per column, half moving first and half second. Sample of %d positions; vs perfect is n/a.\n", *gamesN, t.Decisions)
		} else {
			fmt.Printf("\n%d games per column, half moving first and half second. %s\n", *gamesN, perfectPlay(t))
		}
		return nil
	})
	register("analyze", "why does the brain readout play worse? three experiments", func(args []string) error {
		fs, tree := gameFlags("analyze")
		fs.Parse(args)
		t, err := tree()
		if err != nil {
			return err
		}
		sets := map[string][][]float64{}
		for _, name := range featureset.Names {
			x, err := featureset.Matrix(name, t, featuresPath(t))
			if err != nil {
				return err
			}
			sets[name] = x
		}
		analyze.Run(t, sets)
		return nil
	})
	register("play", "play against the fly in the terminal", func(args []string) error {
		fs, tree := gameFlags("play")
		second := fs.Bool("second", false, "let the fly move first")
		sideFlag := fs.String("side", "", "tic-tac-toe style alias: x moves first, o second")
		live := fs.Bool("live", false, "re-run the full simulation on every fly move")
		fs.Parse(args)
		t, err := tree()
		if err != nil {
			return err
		}
		r, err := loadReadout(t, "brain")
		if err != nil {
			return err
		}
		cache, err := features.Load(featuresPath(t), t)
		if err != nil {
			return err
		}
		brain, eyes, meta, err := loadBrain(t)
		if err != nil {
			return err
		}
		fly := &play.Fly{Readout: r, Cache: cache, Meta: meta}
		if *live || t.Partial {
			fly.Brain, fly.Eyes = brain, eyes
		}
		play.Game(fly, t, !*second && *sideFlag != "o", os.Stdin, os.Stdout)
		return nil
	})
}

// perfectPlay explains what "vs perfect" means for this game.
func perfectPlay(t *game.Tree) string {
	switch t.Value[t.Root] {
	case 0:
		return "The game is a draw under\nperfect play, so 'loss' against the perfect player is how often the player blunders."
	case 1:
		return "The first player wins under\nperfect play, so against the perfect player only the games moving first can be saved."
	default:
		return "The second player wins under\nperfect play, so against the perfect player only the games moving second can be saved."
	}
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
