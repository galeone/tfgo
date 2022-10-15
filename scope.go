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
	"sync/atomic"

	"github.com/galeone/tensorflow/tensorflow/go/op"
)

var tensorCounter uint64

// NewScope returns a unique scope in the format
// input_<suffix> where suffix is a counter
// This function isthread safe can be called in parallel for DIFFERENT scopes.
func NewScope(root *op.Scope) *op.Scope {
	var scope = atomic.AddUint64(&tensorCounter, 1)
	return root.SubScope(fmt.Sprint("input_", scope))
}
