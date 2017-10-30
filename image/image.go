/*
Copyright 2017 Paolo Galeone. All right reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package image

import (
	"fmt"
	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/image/padding"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"strings"
)

// ReadJPEG reads the JPEG image whose path is `imagePath` that has `channels` channels
// it returns an Image
func ReadJPEG(scope *op.Scope, imagePath string, channels int64) *Image {
	scope = tg.NewScope(scope)
	contents := op.ReadFile(scope.SubScope("ReadFile"), op.Const(scope.SubScope("filename"), imagePath))
	output := op.DecodeJpeg(scope.SubScope("DecodeJpeg"), contents, op.DecodeJpegChannels(channels))
	output = op.ExpandDims(scope.SubScope("ExpandDims"), output, op.Const(scope.SubScope("axis"), []int32{0}))
	image := &Image{
		Tensor: tg.NewTensor(scope, output)}
	return image.Scale(0, 1)
}

// ReadPNG reads the PNG image whose path is `imagePath` that has `channels` channels
// it returns an Image
func ReadPNG(scope *op.Scope, imagePath string, channels int64) *Image {
	scope = tg.NewScope(scope)
	contents := op.ReadFile(scope.SubScope("ReadFile"), op.Const(scope.SubScope("filename"), imagePath))
	output := op.DecodePng(scope.SubScope("DecodePng"), contents, op.DecodePngChannels(channels))
	output = op.ExpandDims(scope.SubScope("ExpandDims"), output, op.Const(scope.SubScope("axis"), []int32{0}))
	image := &Image{
		Tensor: tg.NewTensor(scope, output)}
	return image.Scale(0, 1)
}

// ReadGIF reads the GIF image whose path is `imagePath` and returns an Image
func ReadGIF(scope *op.Scope, imagePath string) *Image {
	scope = tg.NewScope(scope)
	contents := op.ReadFile(scope.SubScope("ReadFile"), op.Const(scope.SubScope("filename"), imagePath))
	// DecodeGif returns a Tensor of type uint8. 4-D with shape [num_frames, height, width, 3]. RGB order
	output := op.DecodeGif(scope.SubScope("DecodeGif"), contents)
	image := &Image{
		Tensor: tg.NewTensor(scope, output)}
	return image.Scale(0, 1)
}

// Read search for the `imagePath` extensions and uses the Read<format> function
// to decode and load the right image. Panics if the format is unknown.
func Read(scope *op.Scope, imagePath string, channels int64) *Image {
	split := strings.Split(imagePath, ".")
	ext := strings.ToLower(split[len(split)-1])
	switch ext {
	case "png":
		return ReadPNG(scope, imagePath, channels)
	case "jpg", "jpeg":
		return ReadJPEG(scope, imagePath, channels)
	case "gif":
		return ReadGIF(scope, imagePath)
	default:
		panic(fmt.Errorf("Unsupported image extension %s", ext))
	}
}

// NewImage creates an *Image from a 3 or 4D input tensor
// Place the created image within the specified scope
func NewImage(scope *op.Scope, tensor tf.Output) *Image {
	nd := tensor.Shape().NumDimensions()
	if nd != 3 && nd != 4 {
		panic(fmt.Errorf("tensor should be 3 or 4 D, but has %d dimensions", nd))
	}
	scope = tg.NewScope(scope)
	var output tf.Output
	if nd == 3 {
		output = op.ExpandDims(scope.SubScope("ExpandDims"), tensor, op.Const(scope.SubScope("axis"), []int32{0}))
	} else {
		output = tensor
	}
	// Copy the tensor to a new node in the graph
	output = op.Identity(scope.SubScope("Identity"), output)
	image := &Image{
		Tensor: tg.NewTensor(scope, output)}
	image = image.Scale(0, 1)
	return image
}

// Value returns the 3D tensor that represents a single image
// in the Tensorflow environment.
// If the image is a GIF the returned tensor is 4D
func (image *Image) Value() tf.Output {
	defer image.Tensor.Check()
	if image.Output.Shape().Size(0) == 1 {
		return op.Squeeze(image.Path.SubScope("Squeeze"), image.Output, op.SqueezeSqueezeDims([]int64{0}))
	}
	return image.Output
}

// Scale scales the image range value to be within [min, max]
func (image *Image) Scale(min, max float32) *Image {
	defer image.Tensor.Check()
	if image.Dtype() != tf.Float {
		image = image.Cast(tf.Float)
	}
	s := image.Path.SubScope("scale")
	minVal := op.Min(s.SubScope("Min"), image.Output, op.Const(s.SubScope("reductionIndices"), []int32{0, 1, 2}), op.MinKeepDims(false))
	maxVal := op.Max(s.SubScope("Max"), image.Output, op.Const(s.SubScope("reductionIndices"), []int32{0, 1, 2}), op.MaxKeepDims(false))
	image.Output = op.Div(s.SubScope("Div"),
		op.Mul(s.SubScope("Mul"),
			op.Sub(s.SubScope("Sub"), image.Output, minVal),
			op.Const(s.SubScope("scaleRange"), max-min)),
		op.Sub(s.SubScope("Sub"), maxVal, minVal))
	return image
}

// Normalize computes the mean and the stddev of the pixel values
// and normalizes every pixel subtracting the mean (centering) and dividing by
// the stddev (scale)
func (image *Image) Normalize() *Image {
	defer image.Tensor.Check()
	if image.Dtype() != tf.Float {
		image = image.Cast(tf.Float)
	}

	s := image.Path.SubScope("mean")
	mean := op.Mean(s.SubScope("Mean"), image.Output, op.Const(s.SubScope("reductionIndices"), []int32{0, 1, 2}), op.MeanKeepDims(false))
	s = image.Path.SubScope("variance")
	variance := op.Relu(s.SubScope("Relu"),
		op.Mean(s.SubScope("Mean"),
			op.Sub(s.SubScope("Sub"),
				op.Square(s.SubScope("Square"), image.Output),
				op.Square(s.SubScope("Square"), mean)),
			op.Const(s.SubScope("reductionIndices"), []int32{0, 1, 2})))

	stddev := op.Sqrt(image.Path.SubScope("Sqrt"), variance)

	s = image.Path.SubScope("normalize")
	// Avoid division by zero
	shape := op.Shape(s.SubScope("shape"), image.Output)
	gatherScope := s.SubScope("gatherHeight")
	height := op.Squeeze(s.SubScope("squeeze"), tg.Cast(s, op.Gather(gatherScope, shape, tg.Const(gatherScope.SubScope("indices"), []int32{1})), tf.Float))
	gatherScope = s.SubScope("gatherWidth")
	width := op.Squeeze(s.SubScope("squeeze"), tg.Cast(s, op.Gather(gatherScope, shape, tg.Const(gatherScope.SubScope("indices"), []int32{2})), tf.Float))
	numPixels := op.Mul(s.SubScope("numPix"), width, height)

	minStddev := op.Rsqrt(s.SubScope("Rsqrt"), numPixels)
	pixelValueScale := op.Maximum(s.SubScope("pixveValueScale"), stddev, minStddev)
	image.Output = op.Div(s.SubScope("Div"), op.Sub(s.SubScope("Sub"), image.Output, mean), pixelValueScale)

	return image
}

// Center computes the mean value of the pixel values and subtract this value
// to every pixel: this operation centers the data
func (image *Image) Center() *Image {
	defer image.Tensor.Check()
	if image.Dtype() != tf.Float {
		image = image.Cast(tf.Float)
	}
	s := image.Path.SubScope("center")
	mean := op.Mean(s.SubScope("Mean"), image.Output, op.Const(s.SubScope("reductionIndices"), []int32{0, 1, 2}), op.MeanKeepDims(false))
	image.Output = op.Sub(s.SubScope("Sub"), image.Output, mean)
	return image
}

// SaturateCast casts the image to dtype handling overflow and underflow problems, saturate the exceeding values to
// to minimum/maximum accepted value of the dtype
func (image *Image) SaturateCast(dtype tf.DataType) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("saturateCast")
	if tg.MinValue(image.Dtype()) < tg.MinValue(dtype) {
		image.Output = op.Maximum(s.SubScope("Maximum"), tg.Cast(s, image.Output, tf.Double), op.Const(s.SubScope("Const"), tg.MinValue(dtype)))
	}
	if tg.MaxValue(image.Dtype()) > tg.MaxValue(dtype) {
		image.Output = op.Minimum(s.SubScope("Minimum"), tg.Cast(s, image.Output, tf.Double), op.Const(s.SubScope("Const"), tg.MaxValue(dtype)))
	}
	image = image.Cast(dtype)
	return image
}

// ConvertDtype converts the Image dtype to dtype, uses SaturatesCast if required
func (image *Image) ConvertDtype(dtype tf.DataType, saturate bool) *Image {
	defer image.Tensor.Check()
	if dtype == image.Dtype() {
		return image
	}
	s := image.Path.SubScope("convertDtype")
	if tg.IsInteger(image.Dtype()) && tg.IsInteger(dtype) {
		scaleIn := tg.MaxValue(image.Dtype())
		scaleOut := tg.MaxValue(dtype)
		if scaleIn > scaleOut {
			scale := tg.Cast(s, op.Const(s.SubScope("Const"), int64(scaleIn+1)/int64(scaleOut+1)), image.Dtype())
			image.Output = op.Div(s.SubScope("Div"), image.Output, scale)
			if saturate {
				return image.SaturateCast(dtype)
			}
			image = image.Cast(dtype)
			return image
		}
		if saturate {
			image = image.SaturateCast(dtype)
			scale := tg.Cast(s, op.Const(s.SubScope("Const"), int64(scaleOut+1)/int64(scaleIn+1)), image.Dtype())
			image.Output = op.Mul(s.SubScope("Mul"), image.Output, scale)
		} else {
			image = image.Cast(dtype)
			scale := tg.Cast(s, op.Const(s.SubScope("Const"), int64(scaleOut+1)/int64(scaleIn+1)), image.Dtype())
			image.Output = op.Mul(s.SubScope("Mul"), image.Output, scale)
		}
		return image
	} else if tg.IsFloat(image.Dtype()) && tg.IsFloat(dtype) {
		image = image.Cast(dtype)
		return image
	} else {
		if tg.IsInteger(image.Dtype()) {
			image = image.Cast(dtype)
			scale := tg.Cast(s, op.Const(s.SubScope("Const"), float64(1.0/tg.MaxValue(image.Dtype()))), image.Dtype())
			image.Output = op.Mul(s.SubScope("Mul"), image.Output, scale)
		} else {
			scale := tg.Cast(s, op.Const(s.SubScope("Const"), float64(0.5+tg.MaxValue(dtype))), image.Dtype())
			image.Output = op.Mul(s.SubScope("Mul"), image.Output, scale)
			if saturate {
				return image.SaturateCast(dtype)
			}
			image = image.Cast(dtype)
			return image
		}
	}

	return image
}

// AdjustBrightness adds delta to the image
func (image *Image) AdjustBrightness(delta float32) *Image {
	defer image.Tensor.Check()
	image = image.Add(op.Const(image.Path.SubScope("delta"), delta))
	return image
}

// AdjustContrast changes the contrast by contrastFactor
func (image *Image) AdjustContrast(contrastFactor float32) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("adjustContrast")
	image.Output = op.AdjustContrastv2(s.SubScope("AdjustContrastv2"), image.Output, op.Const(s.SubScope("contrastFactor"), contrastFactor))
	return image
}

// AdjustGamma performs gamma correction on the image
func (image *Image) AdjustGamma(gamma, gain float32) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("adjustGamma")
	dtype := image.Dtype()
	scale := op.Const(s.SubScope("scale"), tg.MaxValue(dtype)-tg.MinValue(dtype))
	scaleTimesGain := op.Const(s.SubScope("scaleTimesGain"), (tg.MaxValue(dtype)-tg.MinValue(dtype))*float64(gain))
	// adjusted_img = (img / scale) ** gamma * scale * gain
	image.Output = op.Mul(s.SubScope("Mul"),
		op.Pow(s.SubScope("Pow"),
			op.Div(s.SubScope("Div"), image.Cast(tf.Double).Output, scale),
			op.Const(s.SubScope("gamma"), float64(gamma))),
		scaleTimesGain)
	return image
}

// AdjustHue changes toe Hue by delta
func (image *Image) AdjustHue(delta float32) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("adjustHue")
	image.Output = op.AdjustHue(s.SubScope("AdjustHue"), image.Output, op.Const(s.SubScope("delta"), delta))
	return image
}

// AdjustSaturation changes the saturation by saturationFactor
func (image *Image) AdjustSaturation(saturationFactor float32) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("adjustSaturation")
	image.Output = op.AdjustSaturation(s.SubScope("AdjustSaturation"), image.Output, op.Const(s.SubScope("saturationFactor"), saturationFactor))
	return image
}

// CentralCrop extracts from the center of the image a portion of image with an area equals to the centralFraction
func (image *Image) CentralCrop(centralFraction float32) *Image {
	defer image.Tensor.Check()

	s := image.Path.SubScope("centralCrop")
	tfCentralFraction := tg.Const(s.SubScope("centralFraction"), centralFraction)

	shape := op.Shape(s.SubScope("shape"), image.Output)
	gatherScope := s.SubScope("gatherHeight")
	height := op.Squeeze(s.SubScope("squeeze"), tg.Cast(s, op.Gather(gatherScope, shape, tg.Const(gatherScope.SubScope("indices"), []int32{1})), tf.Float))
	gatherScope = s.SubScope("gatherWidth")
	width := op.Squeeze(s.SubScope("squeeze"), tg.Cast(s, op.Gather(gatherScope, shape, tg.Const(gatherScope.SubScope("indices"), []int32{2})), tf.Float))

	two := tg.Const(s.SubScope("two"), float32(2))
	hStart := op.Div(s.SubScope("div"),
		op.Sub(s.SubScope("Sub"),
			height,
			op.Mul(s.SubScope("Mul"), height, tfCentralFraction)),
		two)

	wStart := op.Div(s.SubScope("div"),
		op.Sub(s.SubScope("Sub"),
			width,
			op.Mul(s.SubScope("Mul"), width, tfCentralFraction)),
		two)

	hSize := op.Sub(s.SubScope("Sub"), height,
		op.Mul(s.SubScope("Mul"), hStart, two))
	wSize := op.Sub(s.SubScope("Sub"), width,
		op.Mul(s.SubScope("Mul"), wStart, two))

	zero := tg.Const(s.SubScope("zero"), int32(0))
	negOne := tg.Const(s.SubScope("negOne"), int32(-1))

	image.Output = op.Slice(s.SubScope("Slice"), image.Output,
		op.Pack(s.SubScope("begin"), []tf.Output{zero, tg.Cast(s, hStart, tf.Int32), tg.Cast(s, wStart, tf.Int32), zero}),
		op.Pack(s.SubScope("size"), []tf.Output{negOne, tg.Cast(s, hSize, tf.Int32), tg.Cast(s, wSize, tf.Int32), negOne}))
	return image
}

// CropAndResize crops the image to the specified box and resize the result to size
func (image *Image) CropAndResize(box Box, size Size, optional ...op.CropAndResizeAttr) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("cropAndResize")
	boxes := boxes2batch(s, []Box{box})
	image.Output = op.CropAndResize(s.SubScope("CropAndResize"),
		image.Output,
		boxes,
		op.Const(s.SubScope("boxInd"), []int32{0}),
		op.Const(s.SubScope("cropSize"), []int32{int32(size.Height), int32(size.Width)}))
	return image
}

// DrawBoundingBoxes draws the specified boxes to the image
func (image *Image) DrawBoundingBoxes(boxes []Box) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("drawBoundingBoxes")
	image.Output = op.DrawBoundingBoxes(
		s.SubScope("DrawBoundingBoxes"),
		image.Output,
		op.ExpandDims(s.SubScope("ExpandDims"), boxes2batch(s, boxes), op.Const(s.SubScope("axis"), []int32{0})))
	return image
}

// EncodeJPEG encodes the image in the JPEG format and returns an evaluable tensor
func (image *Image) EncodeJPEG(optional ...op.EncodeJpegAttr) tf.Output {
	defer image.Tensor.Check()
	image = image.Scale(0, 255).Cast(tf.Uint8)
	return op.EncodeJpeg(image.Path.SubScope("EncodeJpeg"), image.Value(), optional...)
}

// EncodePNG encodes the image in the PNG format and returns an evaluable tensor
func (image *Image) EncodePNG(optional ...op.EncodePngAttr) tf.Output {
	defer image.Tensor.Check()
	image = image.Scale(0, 255).Cast(tf.Uint8)
	return op.EncodePng(image.Path.SubScope("EncodePng"), image.Value(), optional...)
}

// ExtractGlimpse extracts a glimpse with the specified size at the specified offset
func (image *Image) ExtractGlimpse(size Size, offset Point, optional ...op.ExtractGlimpseAttr) tf.Output {
	defer image.Tensor.Check()
	s := image.Path.SubScope("extractGlimpse")
	offsets := points2batch(s, []Point{offset})
	return op.ExtractGlimpse(s.SubScope("ExtractGlimpse"),
		image.Output,
		op.Const(s.SubScope("size"), []int32{int32(size.Height), int32(size.Width)}), offsets,
		optional...)
}

// RGBToGrayscale converts the image from RGB to Grayscale
func (image *Image) RGBToGrayscale() *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("RGB2Grayscale")
	image = image.Cast(tf.Float)
	rgbWeights := op.Const(s.SubScope("RGBWeights"), []float32{0.2989, 0.5870, 0.1140})
	image.Output = op.Sum(s.SubScope("Sum"), op.Mul(s.SubScope("Mul"), image.Output, rgbWeights), op.Const(s.SubScope("reduction_indices"), []int32{3}), op.SumKeepDims(true))
	return image
}

// HSVToRGB performs the colorspace transformation from HSV to RGB
func (image *Image) HSVToRGB() *Image {
	defer image.Tensor.Check()
	image.Output = op.HSVToRGB(image.Path.SubScope("HSVToRGB"), image.Output)
	return image
}

// RGBToHSV performs the colorspace transformation from RGB to HSV
func (image *Image) RGBToHSV() *Image {
	defer image.Tensor.Check()
	image.Output = op.RGBToHSV(image.Path.SubScope("RGBToHSV"), image.Output)
	return image
}

// ResizeArea resizes the image to the specified size using the Area interpolation
func (image *Image) ResizeArea(size Size, optional ...op.ResizeAreaAttr) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("resizeArea")
	image.Output = op.ResizeArea(s.SubScope("ResizeArea"), image.Output, op.Const(s.SubScope("size"), []int32{int32(size.Height), int32(size.Width)}))
	return image
}

// ResizeBicubic resizes the image to the specified size using the Bicubic interpolation
func (image *Image) ResizeBicubic(size Size, optional ...op.ResizeBicubicAttr) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("resizeBicubic")
	image.Output = op.ResizeBicubic(s.SubScope("ResizeBicubic"), image.Output, op.Const(s.SubScope("size"), []int32{int32(size.Height), int32(size.Width)}))
	return image
}

// ResizeBilinear resizes the image to the specified size using the Bilinear interpolation
func (image *Image) ResizeBilinear(size Size, optional ...op.ResizeBilinearAttr) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("resizeBilinear")
	image.Output = op.ResizeBilinear(s.SubScope("ResizeBilinear"), image.Output, op.Const(s.SubScope("size"), []int32{int32(size.Height), int32(size.Width)}))
	return image
}

// ResizeNearestNeighbor resizes the image to the specified size using the NN interpolation
func (image *Image) ResizeNearestNeighbor(size Size, optional ...op.ResizeNearestNeighborAttr) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("resizeNearestNeighbor")
	image.Output = op.ResizeNearestNeighbor(s.SubScope("ResizeNearestNeighbor"), image.Output, op.Const(s.SubScope("size"), []int32{int32(size.Height), int32(size.Width)}))
	return image
}

// Convolve executes the convolution operation between the current image and the passed filter
// The strides parameter rules the stride along each dimension
// Padding is a padding type to specify the type of padding
func (image *Image) Convolve(filter tf.Output, stride Stride, padding padding.Padding) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("Conv2D")
	// filp the kernel in order to use the correlation operation (here called convolution)
	// like a real convolution operation
	filter = op.ReverseV2(s.SubScope("ReverseV2"), filter, op.Const(s.SubScope("axis"), []int32{0, 1}))
	return image.Correlate(filter, stride, padding)
}

// Correlate executes the correlation operation between the current image and the passed filter
// The strides parameter rules the stride along each dimension
// Padding is a padding type to specify the type of padding
func (image *Image) Correlate(filter tf.Output, stride Stride, padding padding.Padding) *Image {
	defer image.Tensor.Check()
	strides := []int64{1, stride.Y, stride.X, 1}
	image.Output = op.Conv2D(image.Path.SubScope("Corr2D"), image.Output, filter, strides, padding.String())
	return image
}

// Dilate executes the dilatation operation between the current image and the padded filter
// The strides parameter rules the stride along each dimension, in output.
// The rate parameter rules the input stride for atrous morphological dilatation
// Padding is a padding type to specify the type of padding
func (image *Image) Dilate(filter tf.Output, stride, rate Stride, padding padding.Padding) *Image {
	defer image.Tensor.Check()
	strides := []int64{1, stride.Y, stride.X, 1}
	rates := []int64{1, rate.Y, rate.X, 1}
	s := image.Path.SubScope("Dilatation2d")
	// If the filter is a convolutional filter [height, widht, depth, batch]
	// we convert it to a dilatation filter [height, widht, depth]
	if filter.Shape().NumDimensions() == 4 && filter.Shape().Size(3) == 1 {
		filter = op.Squeeze(s, filter, op.SqueezeSqueezeDims([]int64{3}))
	}
	filter = tg.Cast(s, filter, image.Dtype())
	image.Output = op.Dilation2D(s, image.Output, filter, strides, rates, padding.String())
	return image
}

// Erode executes the erosion operation between the current image and the padded filter
// The strides parameter rules the stride along each dimension
// The rate parameter rules the input stride for atrous morphological dilatation
// Padding is a padding type to specify the type of padding
func (image *Image) Erode(filter tf.Output, stride, rate Stride, padding padding.Padding) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("Erode")
	// Negate the input
	negativeOne := tg.Cast(s, op.Const(s.SubScope("negative"), -1.), image.Dtype())
	image = image.Mul(negativeOne)
	// Flip the kernel
	filter = op.ReverseV2(s.SubScope("ReverseV2"), filter, op.Const(s.SubScope("axis"), []int32{0, 1}))
	return image.Dilate(filter, stride, rate, padding)
}
