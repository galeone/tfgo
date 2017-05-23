package image

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Image is the type that abstracts a Tensorflow tensor
// representing an image.
// The image is always a float32 value unless manually casted.
type Image struct {
	value tf.Output
	scope *op.Scope
}

// Point represents a single point in the 2D space
type Point struct {
	X, Y float32
}

// Size represents the spatial extent of an Image
type Size struct {
	Height, Width float32
}

// Rectangle represents a rectangle in the 2D space.
// This rectangle starts from the Point Start and
// has a specified Extent
type Rectangle struct {
	Start  Point
	Extent Size
}

// Box represents the coordinates of 2 points in the space
type Box struct {
	Start, End Point
}

// Stride represents the amount in pixel to move along each dimension
type Stride struct {
	X, Y int64
}

//go:generate enumer -type=Padding types.go
// Padding is an enum to define the type of the padding required
type Padding int

const (
	SAME Padding = iota
	VALID
)
