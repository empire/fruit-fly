# fruit-fly: tic-tac-toe against a simulated fly brain

A learning project in Go built on **MaleCNS v1.0**, the complete wiring diagram of an adult male
*Drosophila melanogaster* central nervous system: 166,700 neurons and 25.6M connections, from
HHMI Janelia and Google Research
([announcement](https://blog.google/innovation-and-ai/technology/research/male-fruit-fly-brain-map/)).

It follows the recipe used by the "a fly plays X" demos (DOOMFLY, flychess,
[flyt3](https://github.com/seanphan/flyt3)) and measures whether the fly brain actually helps.

```
board ──► photoreceptors ──► 25.6M frozen synapses ──► 1,024 descending + ──► linear ──► move
          (R1-R6: mine,       (spiking simulation,       motor neurons         readout
           R8: opponent)       48 ticks)                  (spike counts)        (trained)
```

Only the final linear readout learns. The wiring is never changed.

## Run it

Go 1.27+, CPU only, about 10 minutes end to end on a 16-core laptop. The only dependency is
[arrow-go](https://github.com/apache/arrow-go), for reading the feather files.

```sh
go build -o fly ./cmd/fly
./fly download    # ~1.1 GB into data/raw, 16 parallel range requests (honours HTTPS_PROXY)
./fly build       # feathers -> signed sparse graph (~25 s)
./fly bench       # how fast is one simulation here?
./fly features    # simulate all 4,520 positions once (~5 min), then a sanity check
./fly train       # REINFORCE readouts for brain / random / board features (~15 s)
./fly eval        # results table
./fly analyze     # the three experiments that explain the results (~1 min)
./fly play        # you are X; -side o to go second; -live re-runs the simulation each move
go test ./...
```

## How it works

| stage | package | what happens |
|---|---|---|
| data | `internal/download` | 3 flat tables: neuron annotations, predicted neurotransmitters, connection weights |
| wiring | `internal/connectome` | keep the 166,700 neurons; strength = `log1p(synapses) / sqrt(out-degree+1)`; negative if the sending neuron releases GABA or glutamate |
| eyes and output | `internal/connectome` | all 3,375 R1-R6 photoreceptors are split into 9 groups (one per cell, "my pieces"), and all 1,323 R8 photoreceptors into 9 more ("opponent's pieces"); readout = top 512 descending + top 512 VNC motor neurons |
| brain | `internal/sim` | leaky integrate-and-fire neurons with adaptive thresholds; event-driven (only neurons that just spiked send current); one goroutine per board |
| cache | `internal/features` | the brain is frozen and tic-tac-toe has only 4,520 decision positions, so each is simulated once |
| learning | `internal/readout` | `scores = W · activity + b`, illegal cells masked; REINFORCE with a value baseline against a mix of self, random and perfect opponents; gradients and Adam written by hand and checked by a finite-difference test |
| diagnosis | `internal/analyze` | smoothness, held-out probe, memorization ceiling |
| game | `internal/game`, `internal/play` | rules, minimax, terminal UI |

## Results

`./fly eval` (4,000 games per column, half as X and half as O, greedy moves):

| player | vs random: win | draw | loss | vs perfect: draw | loss |
|---|---|---|---|---|---|
| random moves (baseline) | 44% | 12% | 44% | 13% | 88% |
| **fly brain** (1,024 neurons) | **79%** | 10% | 11% | **57%** | **43%** |
| random ReLU features (1,024, no fly) | 87% | 7% | 7% | 77% | 23% |
| raw board (18 bits) | 87% | 7% | 6% | 93% | 7% |

The fly clearly learns: it beats the random baseline by a wide margin. But a readout on
**random features of the board, the same size and with no fly involved, plays better**, and it
blunders against a perfect player about half as often. Across training seeds the fly-brain readout
lands at 78–83% wins vs random.

`./fly analyze` explains why:

| experiment | fly brain | random features | raw board | chance |
|---|---|---|---|---|
| smoothness: output change from one extra piece ÷ change to an unrelated position (lower = similar boards look similar) | 0.53 | 0.37 | 0.14 | — |
| picks an optimal move on 920 *unseen* positions (linear probe) | 62% | 95% | 69% | 58% |
| picks an optimal move when fitted on *all* positions (memorization) | 75% | 99% | 70% | 58% |

The fly's output does depend on the board: nearly every position gives a distinct spike pattern.
But on unseen positions it's barely better than chance. Similar boards don't look similar enough
in the motor neurons, so the readout has to memorize position by position instead of learning
rules that transfer.

## Lessons from building it

- **The sign rule matters a lot.** With flyt3's rule (only GABA inhibitory, ~87% excitatory), any
  board sets off the same runaway burst through ~47k optic neurons, and the output becomes a
  chaotic hash (smoothness ~0.9). With GABA and glutamate inhibitory (as in Shiu et al. 2024),
  activity flows from the optic lobe to the central brain to the nerve cord. Treating histamine as
  inhibitory too silenced the brain completely, because photoreceptors release histamine.
- **Number precision changed the headline result.** The first version of this project (Python,
  PyTorch, 32-bit floats) reported only 66% wins for the fly. Brain features are spiky: a neuron
  that fires in only a few positions has a standardized value around 60. That makes scores huge,
  and in 32-bit the losing moves' probabilities round to exactly 0, so REINFORCE stops learning
  from them. The same PyTorch code in 64-bit reaches 82%, as does this Go version, which uses
  float64. Other float32 demos may be underreporting for the same reason.
- **Always include a same-size control without the fly.** Without it, "the fly wins 79% of games"
  sounds like the connectome playing tic-tac-toe.

## Honest limits

- **Modelling choices, not biology.** Neuron dynamics, thresholds, the synapse-strength formula
  and the transmitter-to-sign rule are all engineering choices. Real flies also rely on
  neuromodulators, gap junctions and graded potentials, none of which are modelled here.
- **The eye mapping isn't spatial.** The dataset has no retinal coordinates for photoreceptors, so
  board cells go to arbitrary photoreceptor groups.
- **An empty board gives no eye input**, so the fly's opening move as X is fixed by its resting activity.

## Credits

MaleCNS v1.0: HHMI Janelia and Google Research (CC-BY). Recipe: nftechie/doomfly, seanphan/flyt3.
Inhibitory glutamate: Shiu et al., *A Drosophila computational brain model reveals sensorimotor
processing*, Nature 2024.
