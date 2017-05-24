package padding

//go:generate enumer -type=Padding types.go
// Padding is an enum to define the type of the padding required
type Padding int

const (
	SAME Padding = iota
	VALID
)
