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
)

func boxes2batch(scope *op.Scope, boxes []Box) tf.Output {
	s := scope.SubScope("boxes2batch")
	var tfboxes []tf.Output
	for idx, box := range boxes {
		tfboxes = append(tfboxes, op.Const(s.SubScope(fmt.Sprint("idx_", idx)), []float32{box.Start.Y, box.Start.X, box.End.Y, box.End.X}))
	}
	return tfgo.Batchify(s, tfboxes)
}

func sizes2batch(scope *op.Scope, sizes []Size) tf.Output {
	s := scope.SubScope("sizes2batch")
	var tfsizes []tf.Output
	for idx, size := range sizes {
		tfsizes = append(tfsizes, op.Const(s.SubScope(fmt.Sprint("idx_", idx)), []float32{size.Height, size.Width}))
	}
	return tfgo.Batchify(s, tfsizes)
}

func points2batch(scope *op.Scope, points []Point) tf.Output {
	s := scope.SubScope("points2batch")
	var tfpoints []tf.Output
	for idx, point := range points {
		tfpoints = append(tfpoints, op.Const(s.SubScope(fmt.Sprint("idx_", idx)), []float32{point.X, point.Y}))
	}
	return tfgo.Batchify(s, tfpoints)
}
