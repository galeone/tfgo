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

package tfgo_test

import (
	tg "github.com/galeone/tfgo"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"reflect"
	"testing"
)

func TestBatchify(t *testing.T) {
	root := tg.NewRoot()
	var tensors []tf.Output
	for i := 0; i < 10; i++ {
		tensors = append(tensors, tg.Const(root, [3]int32{1, 2, 3}))
	}
	batch := tg.Batchify(root, tensors)

	if batch.Shape().NumDimensions() != 2 {
		t.Errorf("Expected 2D tensor, but got: %dD tensor", batch.Shape().NumDimensions())
	}

	shape, _ := batch.Shape().ToSlice()
	if shape[0] != 10 || shape[1] != 3 {
		t.Errorf("Expected shape (10,3), got  (%d,%d)", shape[0], shape[1])
	}

	result := tg.Exec(root, []tf.Output{batch}, nil, nil)
	// Note the cast to [][] and not to [10][3]
	matrixResult := result[0].Value().([][]int32)
	var expectedMatrix [][]int32
	row := []int32{1, 2, 3}
	for i := 0; i < 10; i++ {
		expectedMatrix = append(expectedMatrix, row)
	}
	if !reflect.DeepEqual(matrixResult, expectedMatrix) {
		t.Errorf("Expected matrix %v\n Got matrix %v", expectedMatrix, matrixResult)
	}
}

func TestIsClose(t *testing.T) {
	root := tg.NewRoot()
	A := tg.Const(root, []float32{0.1, 0.2, 0.3, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 5e-5})
	B := tg.Const(root, []float32{0.11, 0.2, 0.299, 0, 2e-2, 2e-3, 2e-4, 0, 10})
	relTol := tg.Const(root, float32(1e-3))
	absTol := tg.Const(root, float32(1e-6))
	isClose := tg.IsClose(root, A, B, relTol, absTol)

	expected := []bool{false, true, false, false, false, false, false, true, false}
	results := tg.Exec(root, []tf.Output{isClose}, nil, nil)
	result := results[0].Value().([]bool)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected  %v\n Got  %v", expected, result)
	}
}

func TestLoadModel(t *testing.T) {
	model := tg.LoadModel("test_models/export", []string{"tag"}, nil)

	fakeInput, _ := tf.NewTensor([1][28][28][1]float32{})
	results := model.Exec([]tf.Output{
		model.Op("LeNetDropout/softmax_linear/Identity", 0),
	}, map[tf.Output]*tf.Tensor{
		model.Op("input_", 0): fakeInput,
	})

	if results[0].Shape()[0] != 1 || results[0].Shape()[1] != 10 {
		t.Errorf("Expected output shape of [1,10], got %v", results[0].Shape())
	}
}
