/*
Copyright 2017-2022 Paolo Galeone. All right reserved.
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

package filter

import (
	tf "github.com/galeone/tensorflow/tensorflow/go"
	"github.com/galeone/tensorflow/tensorflow/go/op"
)

// arrays & slices are never immutable, so this must be variables
// unexported in order to make then unmodifiable from outside
var (
	sobelX = [3][3][1][1]float32{
		{{{1}}, {{0}}, {{-1}}},
		{{{2}}, {{0}}, {{-2}}},
		{{{1}}, {{0}}, {{-1}}}}
	sobelY = [3][3][1][1]float32{
		{{{1}}, {{2}}, {{1}}},
		{{{0}}, {{0}}, {{0}}},
		{{{-1}}, {{-2}}, {{-1}}}}
)

// SobelX returns a constant tensor with shape [3,3,1,1]
// containing the values of the Sobel operator along X
// Convolving a 2D signal (tensor with shape [height, widht, 1])
// gives as output the directional derivative along the X axis of the signal
func SobelX(scope *op.Scope) tf.Output {
	return op.Const(scope.SubScope("SobelX"), sobelX)
}

// SobelY returns a constant tensor with shape [3,3,1,1]
// containing the values of the Sobel operator along Y
// Convolving a 2D signal (tensor with shape [height, widht, 1])
// gives as output the directional derivative along the Y axis of the signal
func SobelY(scope *op.Scope) tf.Output {
	return op.Const(scope.SubScope("SobelY"), sobelY)
}
