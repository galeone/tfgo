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
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"strings"
)

// ReadJPEG reads the JPEG image whose path is `imagePath` that has `channels` channels
// it returns an Image
func ReadJPEG(scope *op.Scope, imagePath string, channels int64) (image *Image) {
	image = new(Image)
	image.scope = newImageScope(scope)
	contents := op.ReadFile(image.scope.SubScope("ReadFile"), op.Const(image.scope.SubScope("filename"), imagePath))
	image.value = op.DecodeJpeg(image.scope.SubScope("DecodeJpeg"), contents, op.DecodeJpegChannels(channels))
	image.value = op.ExpandDims(image.scope.SubScope("ExpandDims"), image.value, op.Const(image.scope.SubScope("axis"), []int32{0}))
	image = image.Scale(0, 1)
	return image
}

// ReadPNG reads the PNG image whose path is `imagePath` that has `channels` channels
// it returns an Image
func ReadPNG(scope *op.Scope, imagePath string, channels int64) (image *Image) {
	image = new(Image)
	image.scope = newImageScope(scope)
	contents := op.ReadFile(image.scope.SubScope("ReadFile"), op.Const(image.scope.SubScope("filename"), imagePath))
	image.value = op.DecodePng(image.scope.SubScope("DecodePng"), contents, op.DecodePngChannels(channels))
	image.value = op.ExpandDims(image.scope.SubScope("ExpandDims"), image.value, op.Const(image.scope.SubScope("axis"), []int32{0}))
	image = image.Scale(0, 1)
	return image
}

// ReadGIF reads the GIF image whose path is `imagePath` and returns an Image
func ReadGIF(scope *op.Scope, imagePath string) (image *Image) {
	image = new(Image)
	image.scope = newImageScope(scope)
	contents := op.ReadFile(image.scope.SubScope("ReadFile"), op.Const(image.scope.SubScope("filename"), imagePath))
	image.value = op.DecodeGif(image.scope.SubScope("DecodeGif"), contents)
	image.value = op.ExpandDims(image.scope.SubScope("ExpandDims"), image.value, op.Const(image.scope.SubScope("axis"), []int32{0}))
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

// Value returns the 3D tensor that represents a single image
// in the Tensorflow environment.
// If the image is a GIF the returned tensor is 4D
func (image *Image) Value() tf.Output {
	if image.value.Shape().Size(0) == 1 {
		return op.Squeeze(image.scope.SubScope("Squeeze"), image.value, op.SqueezeSqueezeDims([]int64{0}))
	}
	return image.value
}

// Scope returns the scope associated to the image
func (image *Image) Scope() *op.Scope {
	return image.scope
}

// Shape64 returns the shape of the image as []int64.
// If firstDimension is true a 4 elements slice is returned.
// Otherwise a 3 elements sice is returned.
func (image *Image) Shape64(firstDimension bool) []int64 {
	dims64, _ := image.value.Shape().ToSlice()
	if firstDimension {
		return dims64
	}
	return dims64[1:]
}

// Shape32 returns the shape of the image as []int32.
// If firstDimension is true a 4 elements slice is returned.
// Otherwise a 3 elements sice is returned.
func (image *Image) Shape32(firstDimension bool) []int32 {
	dims64 := image.Shape64(firstDimension)
	var dims32 []int32 = make([]int32, len(dims64))
	for idx, dim := range dims64 {
		dims32[idx] = int32(dim)
	}
	return dims32
}

// Clone returns a copy of the current image in a new scope
// Clone must be used when one want to create a different image
// from the output of an operation.
func (image *Image) Clone() *Image {
	clone := new(Image)
	clone.scope = newImageScope(image.scope)
	clone.value = op.Identity(clone.scope.SubScope("Identity"), image.value)
	return clone
}

// Cast casts the current image to the required dtype
func (image *Image) Cast(dtype tf.DataType) *Image {
	image.value = tfgo.Cast(image.scope, image.value, dtype)
	return image
}

// Dtype returns the image dtype
func (image *Image) Dtype() tf.DataType {
	return image.value.DataType()
}

// Add defines the add operation between the image and tensor
// `tensor` dtype is converted to image.Dtype() before adding
func (image *Image) Add(tensor tf.Output) *Image {
	s := image.scope.SubScope("Add")
	image.value = op.Add(s, image.value, tfgo.Cast(s, tensor, image.Dtype()))
	return image
}

// Scale scales the image range value to be within [min, max]
func (image *Image) Scale(min, max float32) *Image {
	if image.Dtype() != tf.Float {
		image = image.Cast(tf.Float)
	}
	s := image.scope.SubScope("scale")
	minVal := op.Min(s.SubScope("Min"), image.value, op.Const(s.SubScope("reductionIndices"), []int32{0, 1, 2}), op.MinKeepDims(false))
	maxVal := op.Max(s.SubScope("Max"), image.value, op.Const(s.SubScope("reductionIndices"), []int32{0, 1, 2}), op.MaxKeepDims(false))
	image.value = op.Div(s.SubScope("Div"),
		op.Mul(s.SubScope("Mul"),
			op.Sub(s.SubScope("Sub"), image.value, minVal),
			op.Const(s.SubScope("scaleRange"), max-min)),
		op.Sub(s.SubScope("Sub"), maxVal, minVal))
	return image
}

// Normalize computes the mean and the stddev of the pixel values
// and normalizes every pixel subtracting the mean (centering) and dividing by
// the stddev (scale)
func (image *Image) Normalize() *Image {
	if image.Dtype() != tf.Float {
		image = image.Cast(tf.Float)
	}

	s := image.scope.SubScope("mean")
	mean := op.Mean(s.SubScope("Mean"), image.value, op.Const(s.SubScope("reductionIndices"), []int32{0, 1, 2}), op.MeanKeepDims(false))
	s = image.scope.SubScope("variance")
	variance := op.Relu(s.SubScope("Relu"),
		op.Mean(s.SubScope("Mean"),
			op.Sub(s.SubScope("Sub"),
				op.Square(s.SubScope("Square"), image.value),
				op.Square(s.SubScope("Square"), mean)),
			op.Const(s.SubScope("reductionIndices"), []int32{0, 1, 2})))

	stddev := op.Sqrt(image.scope.SubScope("Sqrt"), variance)

	s = image.scope.SubScope("normalize")
	// Avoid division by zero
	var numPixels int32
	dims32 := image.Shape32(false)
	for _, dim := range dims32 {
		numPixels *= dim
	}
	minStddev := op.Rsqrt(s.SubScope("Rsqrt"), op.Const(s.SubScope("numPix"), float32(numPixels)))
	pixelValueScale := op.Maximum(s.SubScope("pixveValueScale"), stddev, minStddev)
	image.value = op.Div(s.SubScope("Div"), op.Sub(s.SubScope("Sub"), image.value, mean), pixelValueScale)

	return image
}

// Center computes the mean value of the pixel values and subtract this value
// to every pixel: this operation centers the data
func (image *Image) Center() *Image {
	if image.Dtype() != tf.Float {
		image = image.Cast(tf.Float)
	}
	s := image.scope.SubScope("center")
	mean := op.Mean(s.SubScope("Mean"), image.value, op.Const(s.SubScope("reductionIndices"), []int32{0, 1, 2}), op.MeanKeepDims(false))
	image.value = op.Sub(s.SubScope("Sub"), image.value, mean)
	return image
}

// SaturateCast casts the image to dtype handling overflow and underflow problems, saturate the exceeding values to
// to minimum/maximum accepted value of the dtype
func (image *Image) SaturateCast(dtype tf.DataType) *Image {
	s := image.scope.SubScope("saturateCast")
	if tfgo.MinValue(image.Dtype()) < tfgo.MinValue(dtype) {
		image.value = op.Maximum(s.SubScope("Maximum"), tfgo.Cast(s, image.value, tf.Double), op.Const(s.SubScope("Const"), tfgo.MinValue(dtype)))
	}
	if tfgo.MaxValue(image.Dtype()) > tfgo.MaxValue(dtype) {
		image.value = op.Minimum(s.SubScope("Minimum"), tfgo.Cast(s, image.value, tf.Double), op.Const(s.SubScope("Const"), tfgo.MaxValue(dtype)))
	}
	return image.Cast(dtype)
}

// ConvertDtype converts the Image dtype to dtype, uses SaturatesCast if required
func (image *Image) ConvertDtype(dtype tf.DataType, saturate bool) *Image {
	if dtype == image.Dtype() {
		return image
	}
	s := image.scope.SubScope("convertDtype")
	if tfgo.IsInteger(image.Dtype()) && tfgo.IsInteger(dtype) {
		scaleIn := tfgo.MaxValue(image.Dtype())
		scaleOut := tfgo.MaxValue(dtype)
		if scaleIn > scaleOut {
			scale := op.Const(s.SubScope("Const"), int64(scaleIn+1)/int64(scaleOut+1))
			image.value = op.Div(s.SubScope("Div"), image.value, scale)
			if saturate {
				return image.SaturateCast(dtype)
			} else {
				return image.Cast(dtype)
			}
		} else {
			scale := op.Const(s.SubScope("Const"), int64(scaleOut+1)/int64(scaleIn+1))
			if saturate {
				image = image.SaturateCast(dtype)
				image.value = op.Mul(s.SubScope("Mul"), image.value, scale)
			} else {
				image = image.Cast(dtype)
				image.value = op.Mul(s.SubScope("Mul"), image.value, scale)
			}
			return image
		}
	} else if tfgo.IsFloat(image.Dtype()) && tfgo.IsFloat(dtype) {
		return image.Cast(dtype)
	} else {
		if tfgo.IsInteger(image.Dtype()) {
			image = image.Cast(dtype)
			scale := op.Const(s.SubScope("Const"), float64(1.0/tfgo.MaxValue(image.Dtype())))
			image.value = op.Mul(s.SubScope("Mul"), image.value, scale)
		} else {
			scale := op.Const(s.SubScope("Const"), float64(0.5+tfgo.MaxValue(dtype)))
			image.value = op.Mul(s.SubScope("Mul"), tfgo.Cast(s, image.value, tf.Double), scale)
			if saturate {
				return image.SaturateCast(dtype)
			} else {
				return image.Cast(dtype)
			}
		}
	}

	return image
}

// AdjustBrightness adds delta to the image
func (image *Image) AdjustBrightness(delta float32) *Image {
	return image.Add(op.Const(image.scope.SubScope("delta"), delta))
}

// AdjustContrast changes the contrast by contrastFactor
func (image *Image) AdjustContrast(contrastFactor float32) *Image {
	s := image.scope.SubScope("adjustContrast")
	image.value = op.AdjustContrastv2(s.SubScope("AdjustContrastv2"), image.value, op.Const(s.SubScope("contrastFactor"), contrastFactor))
	return image
}

// AdjustGamma performs gamma correction on the image
func (image *Image) AdjustGamma(gamma, gain float32) *Image {
	s := image.scope.SubScope("adjustGamma")
	dtype := image.Dtype()
	scale := op.Const(s.SubScope("scale"), tfgo.MaxValue(dtype)-tfgo.MinValue(dtype))
	scaleTimesGain := op.Const(s.SubScope("scaleTimesGain"), (tfgo.MaxValue(dtype)-tfgo.MinValue(dtype))*float64(gain))
	// adjusted_img = (img / scale) ** gamma * scale * gain
	image.value = op.Mul(s.SubScope("Mul"), op.Pow(
		s.SubScope("Pow"), op.Div(s.SubScope("Div"), image.Cast(tf.Float).value, scale), op.Const(s.SubScope("gamma"), gamma)),
		tfgo.Cast(s, scaleTimesGain, tf.Float))
	return image
}

// AdjustHue changes toe Hue by delta
func (image *Image) AdjustHue(delta float32) *Image {
	s := image.scope.SubScope("adjustHue")
	image.value = op.AdjustHue(s.SubScope("AdjustHue"), image.value, op.Const(s.SubScope("delta"), delta))
	return image
}

// AdjustSaturation changes the saturation by saturationFactor
func (image *Image) AdjustSaturation(saturationFactor float32) *Image {
	s := image.scope.SubScope("adjustSaturation")
	image.value = op.AdjustSaturation(s.SubScope("AdjustSaturation"), image.value, op.Const(s.SubScope("saturationFactor"), saturationFactor))
	return image
}

// CentralCrop extracts from the center of the image a portion of image with an area equals to the centralFraction
func (image *Image) CentralCrop(centralFraction float32) *Image {
	s := image.scope.SubScope("centralCrop")
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

	image.value = op.Slice(s.SubScope("Slice"), image.value,
		op.Const(s.SubScope("begin"), []float32{0, rect.Start.Y, rect.Start.X, 0}),
		op.Const(s.SubScope("size"), []float32{-1, rect.Extent.Height, rect.Extent.Width, -1}))
	return image
}

// Crop the image to the specified box and resize the result to size
func (image *Image) CropAndResize(box Box, size Size, optional ...op.CropAndResizeAttr) *Image {
	s := image.scope.SubScope("cropAndResize")
	boxes := boxes2batch(s, []Box{box})
	image.value = op.CropAndResize(s.SubScope("CropAndResize"),
		image.value,
		boxes,
		op.Const(s.SubScope("boxInd"), int32(0)),
		op.Const(s.SubScope("cropSize"), []int32{int32(size.Height), int32(size.Width)}))
	return image
}

// DrawBoundingBoxes draws the specified boxes to the image
func (image *Image) DrawBoundingBoxes(boxes []Box) *Image {
	s := image.scope.SubScope("drawBoundingBoxes")
	image.value = op.DrawBoundingBoxes(s.SubScope("DrawBoundingBoxes"), image.value, boxes2batch(s, boxes))
	return image
}

// EncodeJPEG encodes the image in the JPEG format and returns an evaluable tensor
func (image *Image) EncodeJPEG(optional ...op.EncodeJpegAttr) tf.Output {
	image = image.Scale(0, 255).Cast(tf.Uint8)
	return op.EncodeJpeg(image.scope.SubScope("EncodeJpeg"), image.Value(), optional...)
}

// EncodePNG encodes the image in the PNG format and returns an evaluable tensor
func (image *Image) EncodePNG(optional ...op.EncodePngAttr) tf.Output {
	image = image.Scale(0, 255).Cast(tf.Uint8)
	return op.EncodePng(image.scope.SubScope("EncodePng"), image.Value(), optional...)
}

// ExtractGlimpse extracts a set of glimpses with the specified size at the different offests
func (image *Image) ExtractGlimpse(size Size, offsets []Point, optional ...op.ExtractGlimpseAttr) tf.Output {
	s := image.scope.SubScope("extractGlimpse")
	return op.ExtractGlimpse(s.SubScope("ExtractGlimpse"),
		image.value,
		op.Const(s.SubScope("size"), []float32{size.Width, size.Height}), points2batch(s, offsets),
		optional...)
}

// RGB2Grayscale converts the image from RGB to Grayscale
func (image *Image) RGBToGrayscale() *Image {
	s := image.scope.SubScope("RGB2Grayscale")
	image = image.Cast(tf.Float)
	rgbWeights := op.Const(s.SubScope("RGBWeights"), []float32{0.2989, 0.5870, 0.1140})
	image.value = op.Sum(s.SubScope("Sum"), op.Mul(s.SubScope("Mul"), image.value, rgbWeights), op.Const(s.SubScope("reduction_indices"), []int32{3}), op.SumKeepDims(true))
	return image
}

// HSVToRGB performs the colorspace transformation from HSV to RGB
func (image *Image) HSVToRGB() *Image {
	image.value = op.HSVToRGB(image.scope.SubScope("HSVToRGB"), image.value)
	return image
}

// RGBToHSV performs the colorspace transformation from RGB to HSV
func (image *Image) RGBToHSV() *Image {
	image.value = op.RGBToHSV(image.scope.SubScope("RGBToHSV"), image.value)
	return image
}

// ResizeArea resizes the image to the specified size using the Area interpolation
func (image *Image) ResizeArea(size Size, optional ...op.ResizeAreaAttr) *Image {
	s := image.scope.SubScope("resizeArea")
	image.value = op.ResizeArea(s.SubScope("ResizeArea"), image.value, op.Const(s.SubScope("size"), []int32{int32(size.Height), int32(size.Width)}))
	return image
}

// ResizeBicubic resizes the image to the specified size using the Bicubic interpolation
func (image *Image) ResizeBicubic(size Size, optional ...op.ResizeBicubicAttr) *Image {
	s := image.scope.SubScope("resizeBicubic")
	image.value = op.ResizeBicubic(s.SubScope("ResizeBicubic"), image.value, op.Const(s.SubScope("size"), []int32{int32(size.Height), int32(size.Width)}))
	return image
}

// ResizeBilinear resizes the image to the specified size using the Bilinear interpolation
func (image *Image) ResizeBilinear(size Size, optional ...op.ResizeBilinearAttr) *Image {
	s := image.scope.SubScope("resizeBilinear")
	image.value = op.ResizeBilinear(s.SubScope("ResizeBilinear"), image.value, op.Const(s.SubScope("size"), []int32{int32(size.Height), int32(size.Width)}))
	return image
}

// ResizeNearestNeighbor resizes the image to the specified size using the NN interpolation
func (image *Image) ResizeNearestNeighbor(size Size, optional ...op.ResizeNearestNeighborAttr) *Image {
	s := image.scope.SubScope("resizeNearestNeighbor")
	image.value = op.ResizeNearestNeighbor(s.SubScope("ResizeNearestNeighbor"), image.value, op.Const(s.SubScope("size"), []int32{int32(size.Height), int32(size.Width)}))
	return image
}

// Convolve executes the convolution operation between the current image and the passed filter
// The strides parameter rule the stride along each dimension
// Padding is a padding type to specify the type of padding
func (image *Image) Convolve(filter tf.Output, stride Stride, padding Padding) *Image {
	strides := []int64{1, stride.Y, stride.X, 1}
	image.value = op.Conv2D(image.scope.SubScope("Conv2D"), image.value, filter, strides, padding.String())
	return image
}
