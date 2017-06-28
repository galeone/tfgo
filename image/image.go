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
	"github.com/galeone/tfgo"
	"github.com/galeone/tfgo/image/padding"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"strings"
)

// ReadJPEG reads the JPEG image whose path is `imagePath` that has `channels` channels
// it returns an Image
func ReadJPEG(scope *op.Scope, imagePath string, channels int64) (image *Image) {
	image = &Image{Tensor: &tfgo.Tensor{}}
	image.Path = tfgo.NewScope(scope)
	contents := op.ReadFile(image.Path.SubScope("ReadFile"), op.Const(image.Path.SubScope("filename"), imagePath))
	image.Output = op.DecodeJpeg(image.Path.SubScope("DecodeJpeg"), contents, op.DecodeJpegChannels(channels))
	image.Output = op.ExpandDims(image.Path.SubScope("ExpandDims"), image.Output, op.Const(image.Path.SubScope("axis"), []int32{0}))
	image = image.Scale(0, 1)
	return image
}

// ReadPNG reads the PNG image whose path is `imagePath` that has `channels` channels
// it returns an Image
func ReadPNG(scope *op.Scope, imagePath string, channels int64) (image *Image) {
	image = &Image{Tensor: &tfgo.Tensor{}}
	image.Path = tfgo.NewScope(scope)
	contents := op.ReadFile(image.Path.SubScope("ReadFile"), op.Const(image.Path.SubScope("filename"), imagePath))
	image.Output = op.DecodePng(image.Path.SubScope("DecodePng"), contents, op.DecodePngChannels(channels))
	image.Output = op.ExpandDims(image.Path.SubScope("ExpandDims"), image.Output, op.Const(image.Path.SubScope("axis"), []int32{0}))
	image = image.Scale(0, 1)
	return image
}

// ReadGIF reads the GIF image whose path is `imagePath` and returns an Image
func ReadGIF(scope *op.Scope, imagePath string) (image *Image) {
	image = &Image{Tensor: &tfgo.Tensor{}}
	image.Path = tfgo.NewScope(scope)
	contents := op.ReadFile(image.Path.SubScope("ReadFile"), op.Const(image.Path.SubScope("filename"), imagePath))
	image.Output = op.DecodeGif(image.Path.SubScope("DecodeGif"), contents)
	image.Output = op.ExpandDims(image.Path.SubScope("ExpandDims"), image.Output, op.Const(image.Path.SubScope("axis"), []int32{0}))
	image = image.Scale(0, 1)
	return image
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
func NewImage(scope *op.Scope, tensor tf.Output) (image *Image) {
	nd := tensor.Shape().NumDimensions()
	if nd != 3 && nd != 4 {
		panic(fmt.Errorf("tensor should be 3 or 4 D, but has %d dimensions", nd))
	}
	image = &Image{Tensor: &tfgo.Tensor{}}
	image.Path = tfgo.NewScope(scope)
	if nd == 3 {
		image.Output = op.ExpandDims(image.Path.SubScope("ExpandDims"), tensor, op.Const(image.Path.SubScope("axis"), []int32{0}))
	} else {
		image.Output = tensor
	}
	// Copy the tensor to a new node in the graph
	image.Output = op.Identity(image.Path.SubScope("Identity"), image.Output)
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
	var numPixels int32
	dims32 := image.Shape32(false)
	for _, dim := range dims32 {
		numPixels *= dim
	}
	minStddev := op.Rsqrt(s.SubScope("Rsqrt"), op.Const(s.SubScope("numPix"), float32(numPixels)))
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
	if tfgo.MinValue(image.Dtype()) < tfgo.MinValue(dtype) {
		image.Output = op.Maximum(s.SubScope("Maximum"), tfgo.Cast(s, image.Output, tf.Double), op.Const(s.SubScope("Const"), tfgo.MinValue(dtype)))
	}
	if tfgo.MaxValue(image.Dtype()) > tfgo.MaxValue(dtype) {
		image.Output = op.Minimum(s.SubScope("Minimum"), tfgo.Cast(s, image.Output, tf.Double), op.Const(s.SubScope("Const"), tfgo.MaxValue(dtype)))
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
	if tfgo.IsInteger(image.Dtype()) && tfgo.IsInteger(dtype) {
		scaleIn := tfgo.MaxValue(image.Dtype())
		scaleOut := tfgo.MaxValue(dtype)
		if scaleIn > scaleOut {
			scale := op.Const(s.SubScope("Const"), int64(scaleIn+1)/int64(scaleOut+1))
			image.Output = op.Div(s.SubScope("Div"), image.Output, scale)
			if saturate {
				return image.SaturateCast(dtype)
			}
			image = image.Cast(dtype)
			return image
		}
		scale := op.Const(s.SubScope("Const"), int64(scaleOut+1)/int64(scaleIn+1))
		if saturate {
			image = image.SaturateCast(dtype)
			image.Output = op.Mul(s.SubScope("Mul"), image.Output, scale)
		} else {
			image = image.Cast(dtype)
			image.Output = op.Mul(s.SubScope("Mul"), image.Output, scale)
		}
		return image
	} else if tfgo.IsFloat(image.Dtype()) && tfgo.IsFloat(dtype) {
		image = image.Cast(dtype)
		return image
	} else {
		if tfgo.IsInteger(image.Dtype()) {
			image = image.Cast(dtype)
			scale := op.Const(s.SubScope("Const"), float64(1.0/tfgo.MaxValue(image.Dtype())))
			image.Output = op.Mul(s.SubScope("Mul"), image.Output, scale)
		} else {
			scale := op.Const(s.SubScope("Const"), float64(0.5+tfgo.MaxValue(dtype)))
			image.Output = op.Mul(s.SubScope("Mul"), tfgo.Cast(s, image.Output, tf.Double), scale)
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
	scale := op.Const(s.SubScope("scale"), tfgo.MaxValue(dtype)-tfgo.MinValue(dtype))
	scaleTimesGain := op.Const(s.SubScope("scaleTimesGain"), (tfgo.MaxValue(dtype)-tfgo.MinValue(dtype))*float64(gain))
	// adjusted_img = (img / scale) ** gamma * scale * gain
	image.Output = op.Mul(s.SubScope("Mul"), op.Pow(
		s.SubScope("Pow"), op.Div(s.SubScope("Div"), image.Cast(tf.Float).Output, scale), op.Const(s.SubScope("gamma"), gamma)),
		tfgo.Cast(s, scaleTimesGain, tf.Float))
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
	shape := image.Shape32(false)
	//depth := shape[2]
	fractionOffset := int32(1 / ((1 - centralFraction) / 2.0))
	rect := Rectangle{
		Start: Point{
			Y: float32(shape[0] / fractionOffset),
			X: float32(shape[1] / fractionOffset),
		},
		Extent: Size{
			Height: float32(shape[0] - (shape[0]/fractionOffset)*2),
			Width:  float32(shape[1] - (shape[1]/fractionOffset)*2),
		},
	}

	image.Output = op.Slice(s.SubScope("Slice"), image.Output,
		op.Const(s.SubScope("begin"), []float32{0, rect.Start.Y, rect.Start.X, 0}),
		op.Const(s.SubScope("size"), []float32{-1, rect.Extent.Height, rect.Extent.Width, -1}))
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
		op.Const(s.SubScope("boxInd"), int32(0)),
		op.Const(s.SubScope("cropSize"), []int32{int32(size.Height), int32(size.Width)}))
	return image
}

// DrawBoundingBoxes draws the specified boxes to the image
func (image *Image) DrawBoundingBoxes(boxes []Box) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("drawBoundingBoxes")
	image.Output = op.DrawBoundingBoxes(s.SubScope("DrawBoundingBoxes"), image.Output, boxes2batch(s, boxes))
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

// ExtractGlimpse extracts a set of glimpses with the specified size at the different offests
func (image *Image) ExtractGlimpse(size Size, offsets []Point, optional ...op.ExtractGlimpseAttr) tf.Output {
	defer image.Tensor.Check()
	s := image.Path.SubScope("extractGlimpse")
	return op.ExtractGlimpse(s.SubScope("ExtractGlimpse"),
		image.Output,
		op.Const(s.SubScope("size"), []float32{size.Width, size.Height}), points2batch(s, offsets),
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
	filter = tfgo.Cast(s, filter, image.Dtype())
	image.Output = op.Dilation2D(s, image.Output, filter, strides, rates, padding.String())
	return image
}

// Erode ececutes the erosion operation between the current image and the padded filter
// The strides parameter rules the stride along each dimension
// The rate parameter rules the input stride for atrous morphological dilatation
// Padding is a padding type to specify the type of padding
func (image *Image) Erode(filter tf.Output, stride, rate Stride, padding padding.Padding) *Image {
	defer image.Tensor.Check()
	s := image.Path.SubScope("Erode")
	// Negate the input
	negativeOne := tfgo.Cast(s, op.Const(s.SubScope("negative"), -1.), image.Dtype())
	image = image.Mul(negativeOne)
	// Flip the kernel
	filter = op.ReverseV2(s.SubScope("ReverseV2"), filter, op.Const(s.SubScope("axis"), []int32{0, 1}))
	return image.Dilate(filter, stride, rate, padding)
}
