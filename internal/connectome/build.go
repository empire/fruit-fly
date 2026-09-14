// Package connectome turns the MaleCNS v1.0 tables into a signed sparse graph, plus the pools
// of neurons a task can use: photoreceptors to stimulate and motor-side neurons to read out.
//
// The recipe follows nftechie/doomfly and seanphan/flyt3, with one important change:
//   - neurons: every non-glia body with an assigned superclass (drops unassigned fragments)
//   - synapse sign: from the presynaptic neuron's predicted neurotransmitter. GABA and
//     glutamate are inhibitory (as in Shiu et al. 2024), everything else excitatory.
//     flyt3 treats only GABA as inhibitory; with that rule ~87% of neurons excite and any
//     board sets off the same runaway burst. Treating histamine as inhibitory too silences
//     the brain, because photoreceptors release histamine.
//   - strength: log1p(synapse count) / sqrt(out-degree + 1), so hub neurons don't explode
//
// The wiring is never changed afterwards. Nothing about any game is stored in it: package
// retina decides how a game's board is dealt onto the photoreceptor pools.
package connectome

import (
	"cmp"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strings"

	"github.com/empire/fruit-fly/internal/download"
)

const perClass = 512 // read out this many descending neurons and this many VNC motor neurons

// Regions groups neurons for the activity display.
var Regions = [3]string{"optic", "central", "vnc"}

// SensorPoolNames names Graph.SensorPools in order.
var SensorPoolNames = []string{"R1-R6", "R8"}

// Inhibitory lists the neurotransmitters treated as inhibitory.
var Inhibitory = []string{"gaba", "glutamate"}

// Graph is the frozen wiring, stored by presynaptic neuron (outgoing edges):
// neuron i sends Weight[k] to Target[k] for k in Start[i]..Start[i+1].
// This layout suits event-driven simulation: when i spikes, walk its outgoing edges.
type Graph struct {
	N        int
	Start    []int64   // [N+1]
	Target   []int32   // [E]
	Weight   []float32 // [E], signed
	InDegree []float32 // [N]
	Region   []uint8   // [N] index into Regions

	SensorPools [][]int32 // photoreceptor pools, see SensorPoolNames
	Readout     []int32   // 512 descending + 512 VNC motor neurons
}

// Meta describes the graph and names the readout neurons.
type Meta struct {
	Neurons            int      `json:"neurons"`
	Connections        int      `json:"connections"`
	Synapses           int64    `json:"synapses"`
	InhibitoryNT       []string `json:"inhibitory_transmitters"`
	InhibitoryFraction float64  `json:"inhibitory_fraction"`
	ReadoutClasses     []string `json:"readout_classes"`
	ReadoutTypes       []string `json:"readout_types"`
	ReadoutBodyIDs     []int64  `json:"readout_body_ids"`
}

// Build reads the raw tables and writes graph.bin and meta.json to outDir.
func Build(rawDir, outDir string) (*Graph, *Meta, error) {
	ann, err := readFeather(filepath.Join(rawDir, download.Annotations),
		[]string{"bodyId"}, []string{"superclass", "type", "status"})
	if err != nil {
		return nil, nil, err
	}

	// Keep neurons, sorted by body id so we can binary-search ids -> index.
	var keep []int
	for i, sc := range ann.strings["superclass"] {
		if sc != "" && ann.strings["status"][i] != "Glia" {
			keep = append(keep, i)
		}
	}
	slices.SortFunc(keep, func(a, b int) int { return cmp.Compare(ann.ints["bodyId"][a], ann.ints["bodyId"][b]) })
	n := len(keep)
	ids := make([]int64, n)
	superclass := make([]string, n)
	cellType := make([]string, n)
	for k, i := range keep {
		ids[k] = ann.ints["bodyId"][i]
		superclass[k] = ann.strings["superclass"][i]
		cellType[k] = ann.strings["type"][i]
	}
	indexOf := func(body int64) (int, bool) { return slices.BinarySearch(ids, body) }
	fmt.Printf("neurons: %d\n", n)

	nt, err := readFeather(filepath.Join(rawDir, download.Neurotransmitters),
		[]string{"body"}, []string{"consensus_nt"})
	if err != nil {
		return nil, nil, err
	}
	sign := make([]float32, n)
	for i := range sign {
		sign[i] = 1
	}
	for k, body := range nt.ints["body"] {
		if i, ok := indexOf(body); ok && slices.Contains(Inhibitory, strings.ToLower(nt.strings["consensus_nt"][k])) {
			sign[i] = -1
		}
	}

	fmt.Println("reading edges ...")
	edges, err := readFeather(filepath.Join(rawDir, download.Weights),
		[]string{"body_pre", "body_post", "weight"}, nil)
	if err != nil {
		return nil, nil, err
	}
	pre, post, count := edges.ints["body_pre"], edges.ints["body_post"], edges.ints["weight"]

	// Map body ids to indices and drop edges touching non-neurons or self-connections.
	type edge struct {
		from, to int32
		count    int64
	}
	kept := make([]edge, 0, len(pre))
	var synapses int64
	for k := range pre {
		i, ok1 := indexOf(pre[k])
		j, ok2 := indexOf(post[k])
		if ok1 && ok2 && i != j {
			kept = append(kept, edge{int32(i), int32(j), count[k]})
			synapses += count[k]
		}
	}
	edges = nil
	fmt.Printf("connections kept: %d (%d synapses)\n", len(kept), synapses)

	g := &Graph{N: n, Start: make([]int64, n+1), Target: make([]int32, len(kept)),
		Weight: make([]float32, len(kept)), InDegree: make([]float32, n), Region: make([]uint8, n)}
	outDegree := make([]float64, n)
	for _, e := range kept {
		outDegree[e.from]++
		g.InDegree[e.to]++
	}
	// Counting sort by presynaptic neuron, then order targets within each neuron.
	for i := range n {
		g.Start[i+1] = g.Start[i] + int64(outDegree[i])
	}
	fill := slices.Clone(g.Start[:n])
	for _, e := range kept {
		k := fill[e.from]
		fill[e.from]++
		g.Target[k] = e.to
		g.Weight[k] = float32(math.Log1p(float64(e.count)) / math.Sqrt(outDegree[e.from]+1) * float64(sign[e.from]))
	}
	for i := range n {
		lo, hi := g.Start[i], g.Start[i+1]
		idx := make([]int, hi-lo)
		for k := range idx {
			idx[k] = k
		}
		t, w := g.Target[lo:hi], g.Weight[lo:hi]
		slices.SortFunc(idx, func(a, b int) int { return cmp.Compare(t[a], t[b]) })
		ts, ws := slices.Clone(t), slices.Clone(w)
		for k, from := range idx {
			t[k], w[k] = ts[from], ws[from]
		}
	}

	// ---- sensor pools and readout neurons ------------------------------------
	var r16, r8, dn, mn []int32
	for i := range n {
		switch {
		case superclass[i] == "ol_sensory" && cellType[i] == "R1-R6":
			r16 = append(r16, int32(i))
		case superclass[i] == "ol_sensory" && strings.HasPrefix(cellType[i], "R8"):
			r8 = append(r8, int32(i))
		case superclass[i] == "descending_neuron":
			dn = append(dn, int32(i))
		case superclass[i] == "vnc_motor":
			mn = append(mn, int32(i))
		}
		g.Region[i] = region(superclass[i])
	}
	g.SensorPools = [][]int32{r16, r8}
	dn, mn = topByOutDegree(dn, outDegree, perClass), topByOutDegree(mn, outDegree, perClass)
	g.Readout = append(slices.Clone(dn), mn...)

	inhibitory := 0
	for _, s := range sign {
		if s < 0 {
			inhibitory++
		}
	}
	meta := &Meta{Neurons: n, Connections: len(kept), Synapses: synapses, InhibitoryNT: Inhibitory,
		InhibitoryFraction: float64(inhibitory) / float64(n)}
	for k, i := range g.Readout {
		class := "descending_neuron"
		if k >= len(dn) {
			class = "vnc_motor"
		}
		meta.ReadoutClasses = append(meta.ReadoutClasses, class)
		meta.ReadoutTypes = append(meta.ReadoutTypes, cellType[i])
		meta.ReadoutBodyIDs = append(meta.ReadoutBodyIDs, ids[i])
	}

	fmt.Printf("inhibitory neurons (%s): %.1f%%\n", strings.Join(Inhibitory, ", "), 100*meta.InhibitoryFraction)
	fmt.Printf("sensors: %d R1-R6 + %d R8, readout: %d DN + %d motor\n", len(r16), len(r8), len(dn), len(mn))
	return g, meta, Save(outDir, g, meta)
}

func topByOutDegree(neurons []int32, outDegree []float64, n int) []int32 {
	sorted := slices.Clone(neurons)
	slices.SortStableFunc(sorted, func(a, b int32) int { return cmp.Compare(outDegree[b], outDegree[a]) })
	return sorted[:min(n, len(sorted))]
}

func region(superclass string) uint8 {
	switch {
	case strings.HasPrefix(superclass, "ol_") || strings.HasPrefix(superclass, "visual_"):
		return 0
	case strings.HasPrefix(superclass, "vnc_"):
		return 2
	default:
		return 1
	}
}
