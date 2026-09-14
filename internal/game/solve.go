package game

// solve fills Value for every decision node with minimax and marks the optimal moves.
// Finished games must already carry their outcome.
func (t *Tree) solve() {
	const unknown = 2
	for row := range t.Decisions {
		t.Value[row] = unknown
	}
	var value func(node int32) int8
	value = func(node int32) int8 {
		if v := t.Value[node]; v != unknown {
			return v
		}
		best := int8(-2)
		for _, next := range t.Next[node] {
			if next >= 0 {
				best = max(best, -value(next))
			}
		}
		t.Value[node] = best
		return best
	}
	for row := range t.Decisions {
		value(int32(row))
	}
	for row := range t.Decisions {
		t.Best[row] = make([]bool, t.NumActions)
		for a, next := range t.Next[row] {
			t.Best[row][a] = next >= 0 && -t.Value[next] == t.Value[row]
		}
	}
}
