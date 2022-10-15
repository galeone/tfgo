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

package tfgo

import (
	"fmt"
	"math"

	tf "github.com/galeone/tensorflow/tensorflow/go"
)

// IsInteger returns true if dtype is a tensorflow integer type
func IsInteger(dtype tf.DataType) bool {
	switch dtype {
	case tf.Int8, tf.Int16, tf.Int32, tf.Int64, tf.Uint8, tf.Uint16, tf.Qint8, tf.Qint16, tf.Qint32, tf.Quint8, tf.Quint16:
		return true
	default:
		return false
	}
}

// IsFloat returns true if dtype is a tensorfow float type
func IsFloat(dtype tf.DataType) bool {
	switch dtype {
	case tf.Double, tf.Float, tf.Half:
		return true
	default:
		return false
	}
}

// MaxValue returns the maximum value accepted for the dtype
func MaxValue(dtype tf.DataType) float64 {
	switch dtype {
	case tf.Double:
		return math.MaxFloat64
	case tf.Float:
		return math.MaxFloat32
	case tf.Half:
		return math.MaxFloat32 / math.Pow(2, 16)
	case tf.Int16:
		return math.MaxInt16
	case tf.Int32:
		return math.MaxInt32
	case tf.Int64:
		return math.MaxInt64
	case tf.Int8:
		return math.MaxInt8
	case tf.Uint16:
		return math.MaxUint16
	case tf.Uint8:
		return math.MaxUint8
		// No support for Quantized types
	}
	panic(fmt.Sprintf("dtype %d not supported", dtype))
}

// MinValue returns the minimum representable value for the specified dtype
func MinValue(dtype tf.DataType) float64 {
	switch dtype {
	case tf.Double:
		return math.SmallestNonzeroFloat64
	case tf.Float:
		return math.SmallestNonzeroFloat32
	case tf.Half:
		// According to: https://www.khronos.org/opengl/wiki/Small_Float_Formats
		return 6.10 * math.Pow10(-5)
	case tf.Int16:
		return math.MinInt16
	case tf.Int32:
		return math.MinInt32
	case tf.Int64:
		return math.MinInt64
	case tf.Int8:
		return math.MinInt8
	case tf.Uint16, tf.Uint8:
		return 0
		// No support for Quantized types
	}
	panic(fmt.Sprintf("dtype %d not supported", dtype))
}
