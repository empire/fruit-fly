// Package games lists the games the fly can play. Adding a game means implementing
// game.Game in its own package and adding one entry to registry below.
package games

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"github.com/empire/fruit-fly/internal/game"
	"github.com/empire/fruit-fly/internal/games/hexapawn"
	"github.com/empire/fruit-fly/internal/games/tictactoe"
)

type entry struct {
	help string
	// compile builds the game; args is whatever followed "name:" in the spec ("" if nothing).
	compile func(args string) (*game.Tree, error)
}

var registry = map[string]entry{
	"tictactoe": {"3x3 tic-tac-toe", func(args string) (*game.Tree, error) {
		if args != "" {
			return nil, fmt.Errorf("tictactoe takes no size")
		}
		return game.Compile(tictactoe.New())
	}},
	"hexapawn": {"pawns only; default 3 wide x 4 tall, or hexapawn:COLSxROWS", func(args string) (*game.Tree, error) {
		cols, rows := 3, 4
		if args != "" {
			if _, err := fmt.Sscanf(args, "%dx%d", &cols, &rows); err != nil {
				return nil, fmt.Errorf("hexapawn size %q: want COLSxROWS, e.g. 3x4", args)
			}
		}
		g, err := hexapawn.New(cols, rows)
		if err != nil {
			return nil, err
		}
		return game.Compile(g)
	}},
}

// Names lists the registered games with a short description, sorted.
func Names() []string {
	var names []string
	for name, e := range registry {
		names = append(names, fmt.Sprintf("%-10s %s", name, e.help))
	}
	slices.Sort(names)
	return names
}

// Lookup compiles a game from a spec like "tictactoe" or "hexapawn:3x4".
func Lookup(spec string) (*game.Tree, error) {
	name, args, _ := strings.Cut(spec, ":")
	e, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("unknown game %q (see `fly games`)", spec)
	}
	return e.compile(args)
}

// DataDir is where a game's brain cache and trained readouts live.
func DataDir(root string, t *game.Tree) string { return filepath.Join(root, t.Name) }
