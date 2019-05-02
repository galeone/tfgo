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

package tfgo

import (
	"fmt"
	"io/ioutil"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Model represents a trained model
type Model struct {
	saved *tf.SavedModel
}

// LoadModel creates a new *Model, loading it from the exportDir.
// The graph loaded is identified by the set of tags specified when exporting it.
// This operation creates a session with specified `options`
// Panics if the model can't be loaded
func LoadModel(exportDir string, tags []string, options *tf.SessionOptions) (model *Model) {
	var err error
	model = new(Model)
	model.saved, err = tf.LoadSavedModel(exportDir, tags, options)
	if err != nil {
		panic(err.Error())
	}
	return
}

// ImportModel creates a new *Model, loading the graph from the serialized representation.
// This operation creates a session with specified `options`
// Panics if the model can't be loaded
func ImportModel(serializedModel, prefix string, options *tf.SessionOptions) (model *Model) {
	var err error
	model = new(Model)

	defer func() {
		if err != nil {
			panic(err.Error())
		}
	}()

	contents, err := ioutil.ReadFile(serializedModel)
	if err != nil {
		return
	}

	graph := tf.NewGraph()
	if err := graph.Import(contents, prefix); err != nil {
		return
	}

	session, err := tf.NewSession(graph, options)
	if err != nil {
		return
	}

	model.saved = &tf.SavedModel{Session: session, Graph: graph}
	return
}

// Exec executes the nodes/tensors that must be present in the loaded model
// feedDict values to feed to placeholders (that must have been saved in the model definition)
// panics on error
func (model *Model) Exec(tensors []tf.Output, feedDict map[tf.Output]*tf.Tensor) (results []*tf.Tensor) {
	var err error
	if results, err = model.saved.Session.Run(feedDict, tensors, nil); err == nil {
		return results
	}
	panic(err)
}

// Op extracts the output in position idx of the tensor with the specified name from the model graph
func (model *Model) Op(name string, idx int) tf.Output {
	op := model.saved.Graph.Operation(name)
	if op == nil {
		panic(fmt.Errorf("op %s not found", name))
	}
	nout := op.NumOutputs()
	if nout <= idx {
		panic(fmt.Errorf("op %s has %d outputs. Requested output number %d", name, nout, idx))
	}
	return op.Output(idx)
}
