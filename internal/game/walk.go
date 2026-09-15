package game

// Walk is one play of the real rules. Row is the interned decision index, or -1 when
// the position is not in the tree (a sample miss). Off-tree play still follows the rules.
type Walk struct {
	Tree  *Tree
	Row   int
	Legal []int
	Over  bool
	Value int
	First bool

	step   func(action int)
	render []string
	obs    []bool
	parse  func(text string) (int, bool)
	name   func(action int) string
	probs  func(probs []float64) []string
}

// NewWalk starts a game from the initial position.
func (t *Tree) NewWalk() *Walk { return t.newWalk() }

// Step plays a legal action and hands the turn over.
func (w *Walk) Step(action int) { w.step(action) }

// Render draws the current position from a fixed first-player view.
func (w *Walk) Render() []string { return w.render }

// Obs is what the eyes see now.
func (w *Walk) Obs() []bool { return w.obs }

// ParseAction reads a human move at the current position.
func (w *Walk) ParseAction(text string) (int, bool) {
	a, ok := w.parse(text)
	if !ok {
		return 0, false
	}
	for _, legal := range w.Legal {
		if legal == a {
			return a, true
		}
	}
	return 0, false
}

// ActionName writes an action the way a human would type it.
func (w *Walk) ActionName(action int) string { return w.name(action) }

// ProbabilityBoard draws move probabilities, if the game supports it.
func (w *Walk) ProbabilityBoard(probs []float64) ([]string, bool) {
	if w.probs == nil {
		return nil, false
	}
	return w.probs(probs), true
}
