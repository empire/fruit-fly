# fruit-fly

A frozen fly connectome plays board games through a trained linear readout. The README explains
the pipeline and results; this file holds the conventions the code doesn't state.

## Layers

Dependencies point inwards. `game` and `games/*` know nothing about brains; `connectome` and `sim`
know nothing about games; `retina` is the only package that joins them. Code downstream of
`game.Compile` works on a `*game.Tree` and reads action counts, board size and bit counts from it.
A new game is one package in `internal/games/` plus one registry entry, with no edits elsewhere.

## Reproducibility is the refactor test

Every stage is deterministic, so a behaviour-preserving change leaves the data files
**byte-identical**. Verify tic-tac-toe after any change to sim, retina, features, featureset or
readout:

1. Before editing: build the old binary into the scratchpad and copy `data/tictactoe/features.bin`
   and `data/tictactoe/readout_*.bin` next to it.
2. After: `./fly features` (~8 min, run it in the background), `./fly train` (~15 s), then `cmp`
   each file against the copy and `diff` the `./fly eval` output.

Done when every `cmp` is silent. A difference means behaviour changed. Explain it, or find the
cause (RNG call order, float operation order, row order).

- Training is reproducible only at the same `GOMAXPROCS` (one gradient worker per CPU).
- Row order is `Key` order. Random features and the analyze split draw from fixed seeds, so they
  change whenever the row set or the RNG call sequence changes.

## Data and formats

- Per-game files live in `data/<tree.Name>/` (`tictactoe`, `hexapawn-3x4`). `data/raw` and
  `data/processed` are shared by all games.
- A binary format change bumps its magic (`FLYGRAPH2`, `FLYFEAT1`, `FLYREAD1`). Say which
  commands to rerun, e.g. `fly build` for a new graph format.

## Experimental rules

- Report the fly next to both controls (`random` features and the raw `board`), for every game.
- Readout math stays float64. In float32 the probabilities of losing moves round to 0 and
  REINFORCE stops learning from them (see README, "Number precision").
- Supported games are two-player, alternating, perfect-information and enumerable. Simulation costs
  ~65 ms per position on 16 cores; check the position count before choosing a board size.
- When a game has fewer positions than the 1,024 features, `eval` wins can be memorization. Judge
  the fly by the held-out probe in `analyze`.
- Game tests compare position counts and perfect-play values with numbers computed independently
  of the package, never with its own output.
