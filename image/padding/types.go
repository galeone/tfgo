package padding

//go:generate enumer -type=Padding types.go

// Padding is an enum to define the type of the padding required
type Padding int

const (
	// SAME padding causes output shape being equal to input shape
	SAME Padding = iota
	// VALID padding causes output shape being smaller than input shape
	VALID
)
