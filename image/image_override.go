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
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// ------
// ""Overridden"" methods from *Tensor, that in *Tensor
// return *Tensor. Here shoudl change the inner *Tensor
// but returning a *Image value.
// Overridden methods don't need to check for scope error, because
// they use *Tensor methods that already check for this
// ------

// Clone returns a copy of the current image in a new scope
// Clone must be used when one want to create a different image
// from the output of an operation.
func (image *Image) Clone() *Image {
	clone := new(Image)
	clone.Tensor = image.Tensor.Clone()
	return clone
}

// Cast casts the current image tensor to the requested type
func (image *Image) Cast(dtype tf.DataType) *Image {
	image.Tensor = image.Tensor.Cast(dtype)
	return image
}

// Add defines the add operation between the image and tfout
// `tfout` dtype is converted to image.Dtype() before adding
func (image *Image) Add(tfout tf.Output) *Image {
	image.Tensor = image.Tensor.Add(tfout)
	return image
}

// Mul defines the multiplication operation between the tensor
// and `tfout`.
// `tfout` dtype is converted to tensor.Dtype() before multiplying
func (image *Image) Mul(tfout tf.Output) *Image {
	image.Tensor = image.Tensor.Mul(tfout)
	return image
}

// Pow defines the pow operation x^y, where x are the image values
// y dtype is converted to image.Dtype() before executing Pow
func (image *Image) Pow(y tf.Output) *Image {
	image.Tensor = image.Tensor.Pow(y)
	return image
}

// Square defines the square operation for the image values
func (image *Image) Square() *Image {
	image.Tensor = image.Tensor.Square()
	return image
}

// Sqrt defines the square root operation for the image values
func (image *Image) Sqrt() *Image {
	image.Tensor = image.Tensor.Sqrt()
	return image
}
