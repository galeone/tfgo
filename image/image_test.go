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
