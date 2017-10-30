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

package image_test

import (
	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/image"
	"github.com/galeone/tfgo/image/filter"
	"github.com/galeone/tfgo/image/padding"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"reflect"
	"testing"
)

const (
	pngImagePath  string = "./test_images/Enrico_Denti.png"
	jpegImagePath string = "./test_images/angry_pepe.jpg"
	gifImagePath  string = "./test_images/nyanconverted.gif"
)

func TestReadPNG(t *testing.T) {
	root := tg.NewRoot()
	img1 := image.ReadPNG(root, pngImagePath, 3)
	results := tg.Exec(root, []tf.Output{img1.Value()}, nil, &tf.SessionOptions{})
	if !reflect.DeepEqual(results[0].Shape(), []int64{180, 180, 3}) {
		t.Errorf("Expected shape [180, 180, 3] but got %v", results[0].Shape())
	}
	if results[0].DataType() != tf.Float {
		t.Errorf("Expected dtype %d but got %d", tf.Float, results[0].DataType())
	}
}

func TestReadJPEG(t *testing.T) {
	root := tg.NewRoot()
	img1 := image.ReadJPEG(root, jpegImagePath, 3)
	results := tg.Exec(root, []tf.Output{img1.Value()}, nil, &tf.SessionOptions{})
	if !reflect.DeepEqual(results[0].Shape(), []int64{900, 900, 3}) {
		t.Errorf("Expected shape [900, 900, 3] but got %v", results[0].Shape())
	}
	if results[0].DataType() != tf.Float {
		t.Errorf("Expected dtype %d but got %d", tf.Float, results[0].DataType())
	}
}

func TestReadGIF(t *testing.T) {
	root := tg.NewRoot()
	img1 := image.ReadGIF(root, gifImagePath)
	results := tg.Exec(root, []tf.Output{img1.Value()}, nil, &tf.SessionOptions{})
	if !reflect.DeepEqual(results[0].Shape(), []int64{140, 591, 705, 3}) {
		t.Errorf("Expected shape [140, 591, 705, 3] but got %v", results[0].Shape())
	}
	if results[0].DataType() != tf.Float {
		t.Errorf("Expected dtype %d but got %d", tf.Float, results[0].DataType())
	}
}

func TestReadGIFWithRead(t *testing.T) {
	root := tg.NewRoot()
	img1 := image.Read(root, gifImagePath, 3)
	results := tg.Exec(root, []tf.Output{img1.Value()}, nil, &tf.SessionOptions{})
	if !reflect.DeepEqual(results[0].Shape(), []int64{140, 591, 705, 3}) {
		t.Errorf("Expected shape [140, 591, 705, 3] but got %v", results[0].Shape())
	}
	if results[0].DataType() != tf.Float {
		t.Errorf("Expected dtype %d but got %d", tf.Float, results[0].DataType())
	}
}

func TestConvertDtype(t *testing.T) {
	root := tg.NewRoot()
	// default dtype is float with values in 0 1
	img := image.ReadJPEG(root, jpegImagePath, 3)
	unchange := img.Clone().ConvertDtype(tf.Float, false)
	if img.Dtype() != unchange.Dtype() {
		t.Errorf("Dtype shouldn't change but got: %v vs %v", img.Dtype(), unchange.Dtype())
	}

	// convert from float to double
	double := img.Clone().ConvertDtype(tf.Double, false)
	if double.Dtype() != tf.Double {
		t.Errorf("Double type expected but got: %v", double.Dtype())
	}

	// Float to int, no saturate
	intNoSat := img.Clone().ConvertDtype(tf.Int32, false)
	if intNoSat.Dtype() != tf.Int32 {
		t.Errorf("Expected int32, but got: %v", intNoSat.Dtype())
	}

	// Float to int, saturate
	intSat := img.Clone().ConvertDtype(tf.Int32, true)
	if intSat.Dtype() != tf.Int32 {
		t.Errorf("Expected int32, but got: %v", intSat.Dtype())
	}

	// From int to bigger int, with saturate
	longInt := intSat.Clone().ConvertDtype(tf.Int64, true)
	if longInt.Dtype() != tf.Int64 {
		t.Errorf("Expected int64, but got: %v", longInt.Dtype())
	}

	longIntNoSat := intSat.Clone().ConvertDtype(tf.Int64, false)
	if longIntNoSat.Dtype() != tf.Int64 {
		t.Errorf("Expected int64, but got: %v", longIntNoSat.Dtype())
	}
	// From int to smaller int
	shortInt := intSat.Clone().ConvertDtype(tf.Int8, true)
	if shortInt.Dtype() != tf.Int8 {
		t.Errorf("Expected int8, but got: %v", shortInt.Dtype())
	}

	shortIntNoSat := intSat.Clone().ConvertDtype(tf.Int8, false)
	if shortIntNoSat.Dtype() != tf.Int8 {
		t.Errorf("Expected int8, but got: %v", shortIntNoSat.Dtype())
	}

	// From int to float, saturate or not is meaningless
	floatFromInt := intSat.Clone().ConvertDtype(tf.Double, false)
	if floatFromInt.Dtype() != tf.Double {
		t.Errorf("Expected tf.double, got %v", floatFromInt.Dtype())
	}
}

func TestChangeColorspace(t *testing.T) {
	root := tg.NewRoot()
	img := image.ReadJPEG(root, jpegImagePath, 3)
	imgToHSV := img.Clone().RGBToHSV()
	imgToRGB := imgToHSV.Clone().HSVToRGB()
	// isClose returns the comparison elementwhise among two values
	// I want a single value -> put them all in and
	closeness := op.All(root.SubScope("All"),
		tg.IsClose(root,
			img.Value(), imgToRGB.Value(), tg.Const(root, float32(1e-5)), tg.Const(root, float32(1e-7))),
		tg.Const(root.SubScope("reduction_indices"), []int32{0, 1, 2}))
	results := tg.Exec(root, []tf.Output{closeness}, nil, &tf.SessionOptions{})
	if !results[0].Value().(bool) {
		t.Error("RGB -> HSB -> RGB expect to have first RGB equal to second RGB, but they're different")
	}
}

func TestPanicReadWithUnsupportedExtensaion(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Code did not panic, but it should")
		}
	}()
	root := tg.NewRoot()
	image.Read(root, "not_exists.jlel", 1)
}

func TestNewImage(t *testing.T) {
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3)

	// img is a 4-D tensor under the hood
	clone := image.NewImage(root, img.Output)
	// Extract 4d tensor, remove batch size, use NewImage -> adds first dim -> 4d
	clone3d := image.NewImage(root, op.Squeeze(root.SubScope("Squeeze"), clone.Output, op.SqueezeSqueezeDims([]int64{0})))
	if !reflect.DeepEqual(clone.Shape64(true), clone3d.Shape64(true)) {
		t.Errorf("clone shape = %v must be equal to %v shape, but is not", clone.Shape64(true), clone3d.Shape64(true))
	}
}

func TestPanicNewImage(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Code did not panic, but it should")
		}
	}()
	// a new image is meanigless if its not a 3 or 4d tensor
	root := tg.NewRoot()
	tensor := tg.Const(root, []uint32{1})
	image.NewImage(root, tensor)
}

func TestResizeArea(t *testing.T) {
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3)
	resize1 := img.Clone().ResizeArea(image.Size{Height: 80, Width: 80}).Value()
	resize2 := img.Clone().ResizeArea(image.Size{Height: 30, Width: 30}).Value()

	results := tg.Exec(root, []tf.Output{resize1, resize2}, nil, &tf.SessionOptions{})
	if !reflect.DeepEqual(results[0].Shape(), []int64{80, 80, 3}) {
		t.Errorf("Expected shape [80, 80, 3] but got %v", results[0].Shape())
	}
	if !reflect.DeepEqual(results[1].Shape(), []int64{30, 30, 3}) {
		t.Errorf("Expected shape [30, 30, 3] but got %v", results[0].Shape())
	}
}

func TestResizeBicubic(t *testing.T) {
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3)
	resize1 := img.Clone().ResizeBicubic(image.Size{Height: 80, Width: 80}).Value()
	resize2 := img.Clone().ResizeBicubic(image.Size{Height: 30, Width: 30}).Value()

	results := tg.Exec(root, []tf.Output{resize1, resize2}, nil, &tf.SessionOptions{})
	if !reflect.DeepEqual(results[0].Shape(), []int64{80, 80, 3}) {
		t.Errorf("Expected shape [80, 80, 3] but got %v", results[0].Shape())
	}
	if !reflect.DeepEqual(results[1].Shape(), []int64{30, 30, 3}) {
		t.Errorf("Expected shape [30, 30, 3] but got %v", results[0].Shape())
	}
}

func TestResizeBilinear(t *testing.T) {
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3)
	resize1 := img.Clone().ResizeBilinear(image.Size{Height: 80, Width: 80}).Value()
	resize2 := img.Clone().ResizeBilinear(image.Size{Height: 30, Width: 30}).Value()

	results := tg.Exec(root, []tf.Output{resize1, resize2}, nil, &tf.SessionOptions{})
	if !reflect.DeepEqual(results[0].Shape(), []int64{80, 80, 3}) {
		t.Errorf("Expected shape [80, 80, 3] but got %v", results[0].Shape())
	}
	if !reflect.DeepEqual(results[1].Shape(), []int64{30, 30, 3}) {
		t.Errorf("Expected shape [30, 30, 3] but got %v", results[0].Shape())
	}
}

func TestResizeNearestNeighbor(t *testing.T) {
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3)
	resize1 := img.Clone().ResizeNearestNeighbor(image.Size{Height: 80, Width: 80}).Value()
	resize2 := img.Clone().ResizeNearestNeighbor(image.Size{Height: 30, Width: 30}).Value()

	results := tg.Exec(root, []tf.Output{resize1, resize2}, nil, &tf.SessionOptions{})
	if !reflect.DeepEqual(results[0].Shape(), []int64{80, 80, 3}) {
		t.Errorf("Expected shape [80, 80, 3] but got %v", results[0].Shape())
	}
	if !reflect.DeepEqual(results[1].Shape(), []int64{30, 30, 3}) {
		t.Errorf("Expected shape [30, 30, 3] but got %v", results[0].Shape())
	}
}

func TestAdd(t *testing.T) {
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3).ResizeNearestNeighbor(image.Size{Height: 80, Width: 80}).Cast(tf.Double)
	dims := img.Shape32(false)
	s := img.Scope()
	noise := op.ParameterizedTruncatedNormal(s.SubScope("ParameterizedTruncatedNormal"),
		op.Const(s.SubScope("shape"), dims),
		op.Const(s.SubScope("means"), 0.),
		op.Const(s.SubScope("stddev"), 1.),
		op.Const(s.SubScope("minvals"), 0.),
		op.Const(s.SubScope("maxvals"), 1.))
	noisyImage := img.Clone().Add(noise).Cast(tf.Double)

	results := tg.Exec(root, []tf.Output{img.Value(), noise, noisyImage.Value()}, nil, &tf.SessionOptions{})
	floatImg := results[0].Value().([][][]float64)
	floatNoise := results[1].Value().([][][]float64)
	floatNoisyImage := results[2].Value().([][][]float64)

	if floatNoisyImage[0][0][0] != (floatImg[0][0][0] + floatNoise[0][0][0]) {
		t.Errorf("Add img + noise should be coherent but got: %f != %f + %f", floatNoisyImage[0][0][0], floatImg[0][0][0], floatNoise[0][0][0])
	}
}

func TestConvolveCorrelate(t *testing.T) {
	root := tg.NewRoot()
	grayImg := image.Read(root, pngImagePath, 1)
	grayImg = grayImg.Scale(0, 255)

	// Edge detection using sobel filter: convolution
	Gx := grayImg.Clone().Convolve(filter.SobelX(root), image.Stride{X: 1, Y: 1}, padding.SAME)
	Gy := grayImg.Clone().Convolve(filter.SobelY(root), image.Stride{X: 1, Y: 1}, padding.SAME)
	// *image.Value -> revemo batch size if = 1
	convoluteEdges := image.NewImage(root.SubScope("edge"), Gx.Square().Add(Gy.Square().Value()).Sqrt().Value()).Value()

	// correlation
	Gx = grayImg.Clone().Correlate(filter.SobelX(root), image.Stride{X: 1, Y: 1}, padding.VALID)
	Gy = grayImg.Clone().Correlate(filter.SobelY(root), image.Stride{X: 1, Y: 1}, padding.VALID)
	correlateEdges := image.NewImage(root.SubScope("edge"), Gx.Pow(tg.Const(root, int32(2))).Add(Gy.Square().Value()).Sqrt().Value()).Value()

	results := tg.Exec(root, []tf.Output{convoluteEdges, correlateEdges}, nil, &tf.SessionOptions{})
	if !reflect.DeepEqual(results[0].Shape(), []int64{180, 180, 1}) {
		t.Errorf("Expected shape [180, 180, 1] but got %v", results[0].Shape())
	}
	// Same padding -> equal size
	if !reflect.DeepEqual(results[0].Shape(), []int64{180, 180, 1}) {
		t.Errorf("Convolution with SAME padding should produce output shape = input shape, but got: %v != %v", results[0].Shape(), []int64{180, 180, 1})
	}

	// Valid padding -> output shape < input shape
	if reflect.DeepEqual(results[1], []int64{180, 180, 1}) {
		t.Errorf("Convolution/correlation with VALID padding should produce output shape < input shape, but they are equal to: %v", results[1].Shape())
	}

}

func TestAdjustBrightness(t *testing.T) {
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3).ResizeNearestNeighbor(image.Size{Height: 80, Width: 80})
	imgBright := img.Clone().AdjustBrightness(0.5)
	results := tg.Exec(root, []tf.Output{img.Value(), imgBright.Value()}, nil, &tf.SessionOptions{})
	originalImg := results[0].Value().([][][]float32)
	brightImg := results[1].Value().([][][]float32)
	if originalImg[2][2][2]+0.5 != brightImg[2][2][2] {
		t.Errorf("AjustBrghtness expect to add delta=0.5 to %f but got %f", originalImg[2][2][2], brightImg[2][2][2])
	}
}

func TestAdjustContrast(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Code panic, but it shouldn't: %v", r)
		}
	}()
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3).ResizeNearestNeighbor(image.Size{Height: 80, Width: 80})
	imgContrast := img.Clone().AdjustContrast(0.5)
	tg.Exec(root, []tf.Output{imgContrast.Value()}, nil, &tf.SessionOptions{})
	// If no panic, tensorflow works and the change in contrast exists
}

func TestAdjustGamma(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Code panic, but it shouldn't: %v", r)
		}
	}()
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3).ResizeNearestNeighbor(image.Size{Height: 80, Width: 80})
	imgGamma := img.Clone().AdjustGamma(0.5, 0.7)
	tg.Exec(root, []tf.Output{imgGamma.Value()}, nil, &tf.SessionOptions{})
	// If no panic, tensorflow works and the change in contrast exists
}

func TestAdjustHue(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Code panic, but it shouldn't: %v", r)
		}
	}()
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3).ResizeNearestNeighbor(image.Size{Height: 80, Width: 80})
	imgHue := img.Clone().AdjustHue(0.5)
	tg.Exec(root, []tf.Output{imgHue.Value()}, nil, &tf.SessionOptions{})
	// If no panic, tensorflow works and the change in contrast exists
}

func TestAdjustSaturation(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Code panic, but it shouldn't: %v", r)
		}
	}()
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3).ResizeNearestNeighbor(image.Size{Height: 80, Width: 80})
	imgSaturation := img.Clone().AdjustSaturation(0.5)
	tg.Exec(root, []tf.Output{imgSaturation.Value()}, nil, &tf.SessionOptions{})
	// If no panic, tensorflow works and the change in contrast exists
}

func TestEncodeJPEG(t *testing.T) {
	root := tg.NewRoot()
	img := image.Read(root, jpegImagePath, 3)
	results := tg.Exec(root, []tf.Output{img.EncodeJPEG()}, nil, &tf.SessionOptions{})
	if len(results[0].Value().(string)) == 0 {
		t.Error("Encoding of a just read image should produce an image without any problem")
	}
}

func TestEncodePNG(t *testing.T) {
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3)
	results := tg.Exec(root, []tf.Output{img.EncodePNG()}, nil, &tf.SessionOptions{})
	if len(results[0].Value().(string)) == 0 {
		t.Error("Encoding of a just read image should produce an image without any problem")
	}
}

func TestRGB2Grayscale(t *testing.T) {
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 3).ResizeArea(image.Size{Height: 80, Width: 80}).RGBToGrayscale().Value()
	results := tg.Exec(root, []tf.Output{img}, nil, &tf.SessionOptions{})

	if !reflect.DeepEqual(results[0].Shape(), []int64{80, 80, 1}) {
		t.Errorf("Expected [80, 80, 1] but got %v", results[0].Shape())
	}
}

func TestCropAndResize(t *testing.T) {
	root := tg.NewRoot()
	crop := image.Read(root, pngImagePath, 3).CropAndResize(image.Box{Start: image.Point{X: 0.2, Y: 0.2},
		End: image.Point{X: 0.5, Y: 0.8}},
		image.Size{Height: 200, Width: 200})

	results := tg.Exec(root, []tf.Output{crop.Value()}, nil, &tf.SessionOptions{})
	if !reflect.DeepEqual(results[0].Shape(), []int64{200, 200, 3}) {
		t.Errorf("Expected the crop and resize with shape 200x200, but got: %v", results[0].Shape())
	}
}

func TestDrawBB(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Code panic, but it shouldn't: %v", r)
		}
	}()
	root := tg.NewRoot()
	singleBox := image.Read(root, pngImagePath, 3).DrawBoundingBoxes([]image.Box{{
		Start: image.Point{X: 0.2, Y: 0.2},
		End:   image.Point{X: 0.5, Y: 0.8}}})

	doubleBox := image.Read(root, pngImagePath, 3).DrawBoundingBoxes([]image.Box{{
		Start: image.Point{X: 0.2, Y: 0.2},
		End:   image.Point{X: 0.5, Y: 0.8}}, {
		Start: image.Point{X: 0.05, Y: 0.27},
		End:   image.Point{X: 0.7, Y: 0.9}}})

	// If no panic is thrown, we suppose that tensorflow works and drawed the boxes
	tg.Exec(root, []tf.Output{singleBox.Value(), doubleBox.Value()}, nil, &tf.SessionOptions{})
}

func TestExtractGlimpses(t *testing.T) {
	root := tg.NewRoot()
	glimpse := image.Read(root, pngImagePath, 3).ExtractGlimpse(image.Size{Height: 200, Width: 300},
		image.Point{X: 0.2, Y: 0.2}, op.ExtractGlimpseNormalized(true), op.ExtractGlimpseCentered(false))
	// normalized = true and centered = false -> coordinates in [0,1]
	results := tg.Exec(root, []tf.Output{glimpse}, nil, &tf.SessionOptions{})
	glimpses := results[0]
	if !reflect.DeepEqual(glimpses.Shape(), []int64{1, 200, 300, 3}) {
		t.Errorf("Expected 1 glimpse 200x300x3, got: %v", glimpses.Shape())
	}
}

func TestCentralCrop(t *testing.T) {
	root := tg.NewRoot()
	centralCrop := image.Read(root, pngImagePath, 3).CentralCrop(0.5).Value()
	results := tg.Exec(root, []tf.Output{centralCrop}, nil, &tf.SessionOptions{})
	if !reflect.DeepEqual(results[0].Shape(), []int64{90, 90, 3}) {
		t.Errorf("Expected a central crop of 50%% (180x180) -> 90x90, but got: %v", results[0].Shape())
	}
}

func TestDilateErode(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Code panic, but it shouldn't: %v", r)
		}
	}()
	root := tg.NewRoot()
	// image 1D, ok filter 1d
	img := image.Read(root, pngImagePath, 1)
	dilate := img.Clone().Dilate(filter.SobelX(root), image.Stride{X: 1, Y: 1}, image.Stride{X: 2, Y: 2}, padding.SAME).Value()
	erode := img.Clone().Erode(filter.SobelX(root), image.Stride{X: 1, Y: 1}, image.Stride{X: 2, Y: 2}, padding.SAME).Value()
	tg.Exec(root, []tf.Output{dilate, erode}, nil, &tf.SessionOptions{})
}

func TestPanicMorph(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Code did not panic, but it should")
		}
	}()
	root := tg.NewRoot()
	// image 3D, filter 1d -> panic
	img := image.Read(root, pngImagePath, 3)
	dilate := img.Clone().Dilate(filter.SobelX(root), image.Stride{X: 1, Y: 1}, image.Stride{X: 2, Y: 2}, padding.SAME).Value()
	tg.Exec(root, []tf.Output{dilate}, nil, &tf.SessionOptions{})
}

func TestCenterNormalized(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Code panic, but it shouldn't: %v", r)
		}
	}()
	root := tg.NewRoot()
	img := image.Read(root, pngImagePath, 1)
	normalized := img.Clone().Normalize().Value()
	centered := img.Clone().Center().Value()
	tg.Exec(root, []tf.Output{normalized, centered}, nil, &tf.SessionOptions{})
}
