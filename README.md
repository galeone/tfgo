# tfgo: Tensorflow in Go
[![GoDoc](https://godoc.org/github.com/galeone/tfgo?status.svg)](https://godoc.org/github.com/galeone/tfgo)
[![Build Status](https://travis-ci.org/galeone/tfgo.svg?branch=master)](https://travis-ci.org/galeone/tfgo)
---

Tensorflow's Go bindings are [hard to use](https://pgaleone.eu/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/): tfgo makes it easy!

No more problems like:

- Scoping: each new node will have a new and unique name
- Typing: attributes are automatically converted to a supported type instead of throwing errors at runtime

Also, it uses [Method chaining](https://en.wikipedia.org/wiki/Method_chaining) making possible to write pleasant Go code.

# Getting started

Prerequisite: https://www.tensorflow.org/versions/master/install/install_go

The core data structure of the Tensorflow's Go bindings is the `op.Scope` struct. tfgo allows creating new `*op.Scope` that solves the scoping issue mentioned above.

Since we're defining a graph, let's start from its root (empty graph)

```go
root := tg.NewRoot()
```

We can now place nodes into this graphs and connect them. Let's say we want to multiply a matrix for a column vector and then add another column vector to the result.

Here's the complete source code.

```go
package main

import (
        "fmt"
        tg "github.com/galeone/tfgo"
        tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
        root := tg.NewRoot()
        A := tg.NewTensor(root, tg.Const(root, [2][2]int32{{1, 2}, {-1, -2}}))
        x := tg.NewTensor(root, tg.Const(root, [2][1]int64{{10}, {100}}))
        b := tg.NewTensor(root, tg.Const(root, [2][1]int32{{-10}, {10}}))
        Y := A.MatMul(x.Output).Add(b.Output)
        // Please note that Y is just a pointer to A!

        // If we want to create a different node in the graph, we have to clone Y
        // or equivalently A
        Z := A.Clone()
        results := tg.Exec(root, []tf.Output{Y.Output, Z.Output}, nil, &tf.SessionOptions{})
        fmt.Println("Y: ", results[0].Value(), "Z: ", results[1].Value())
        fmt.Println("Y == A", Y == A) // ==> true
        fmt.Println("Z == A", Z == A) // ==> false
}
```
that produces
```
Y:  [[200] [-200]] Z:  [[200] [-200]]
Y == A true
Z == A false
```

The list of the available methods is available on GoDoc: http://godoc.org/github.com/galeone/tfgo

## Computer Vision using data flow graph

Tensorflow is rich of methods for performing operations on images. tfgo provides the `image` package that allows using the Go bindings to perform computer vision tasks in an elegant way.

For instance, it's possible to read an image, compute its directional derivative along the horizontal and vertical directions, compute the gradient and save it.

The code below does that, showing the different results achieved using correlation and convolution operations.

```go
package main

import (
        tg "github.com/galeone/tfgo"
        "github.com/galeone/tfgo/image"
        "github.com/galeone/tfgo/image/filter"
        "github.com/galeone/tfgo/image/padding"
        tf "github.com/tensorflow/tensorflow/tensorflow/go"
        "os"
)

func main() {
        root := tg.NewRoot()
        grayImg := image.Read(root, "/home/pgaleone/airplane.png", 1)
        grayImg = grayImg.Scale(0, 255)

        // Edge detection using sobel filter: convolution
        Gx := grayImg.Clone().Convolve(filter.SobelX(root), image.Stride{X: 1, Y: 1}, padding.SAME)
        Gy := grayImg.Clone().Convolve(filter.SobelY(root), image.Stride{X: 1, Y: 1}, padding.SAME)
        convoluteEdges := image.NewImage(root.SubScope("edge"), Gx.Square().Add(Gy.Square().Value()).Sqrt().Value()).EncodeJPEG()

        Gx = grayImg.Clone().Correlate(filter.SobelX(root), image.Stride{X: 1, Y: 1}, padding.SAME)
        Gy = grayImg.Clone().Correlate(filter.SobelY(root), image.Stride{X: 1, Y: 1}, padding.SAME)
        correlateEdges := image.NewImage(root.SubScope("edge"), Gx.Square().Add(Gy.Square().Value()).Sqrt().Value()).EncodeJPEG()

        results := tg.Exec(root, []tf.Output{convoluteEdges, correlateEdges}, nil, &tf.SessionOptions{})

        file, _ := os.Create("convolved.png")
        file.WriteString(results[0].Value().(string))
        file.Close()

        file, _ = os.Create("correlated.png")
        file.WriteString(results[1].Value().(string))
        file.Close()
}

```

**airplane.png**

![airplane](https://i.imgur.com/QS6shgc.jpg)

**convolved.png**

![convolved](https://i.imgur.com/zVndo9B.jpg)

**correlated.png**

![correlated](https://i.imgur.com/vhYF7o3.jpg)

the list of the available methods is available on GoDoc: http://godoc.org/github.com/galeone/tfgo/image

## Train in Python, Serve in Go

Using both [DyTB](https://github.com/galeone/dynamic-training-bench) and tfgo we can train, evaluate and export a machine learning model in very few lines of Python and Go code. Below you can find the Python and the Go code.
Just dig into the example to understand how to serve a trained model with `tfgo`.

### Python code

```python
import sys
import tensorflow as tf
from dytb.inputs.predefined.MNIST import MNIST
from dytb.models.predefined.LeNetDropout import LeNetDropout
from dytb.train import train

def main():
    """main executes the operations described in the module docstring"""
    lenet = LeNetDropout()
    mnist = MNIST()

    info = train(
        model=lenet,
        dataset=mnist,
        hyperparameters={"epochs": 2},)

    checkpoint_path = info["paths"]["best"]

    with tf.Session() as sess:
        # Define a new model, import the weights from best model trained
        # Change the input structure to use a placeholder
        images = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="input_")
        # define in the default graph the model that uses placeholder as input
        _ = lenet.get(images, mnist.num_classes)

        # The best checkpoint path contains just one checkpoint, thus the last is the best
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

        # Create a builder to export the model
        builder = tf.saved_model.builder.SavedModelBuilder("export")
        # Tag the model in order to be capable of restoring it specifying the tag set
        # clear_device=True in order to export a device agnostic graph.
        builder.add_meta_graph_and_variables(sess, ["tag"], clear_devices=True)
        builder.save()

    return 0


if __name__ == '__main__':
    sys.exit(main())
```

### Go code

```go
package main

import (
        "fmt"
        tg "github.com/galeone/tfgo"
        tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
        model := tg.LoadModel("test_models/export", []string{"tag"}, nil)

        fakeInput, _ := tf.NewTensor([1][28][28][1]float32{})
        results := model.Exec([]tf.Output{
                model.Op("LeNetDropout/softmax_linear/Identity", 0),
        }, map[tf.Output]*tf.Tensor{
                model.Op("input_", 0): fakeInput,
        })

        predictions := results[0].Value().([][]float32)
        fmt.Println(predictions)
}
```

## Train with tf.estimator, serve in go

`tfgo` supports two different inputs for the estimator:

- Pandas DataFrames
- Numpy Arrays
- Python Dictionary

You can train you estimator using these three types of feature columns and you'll be able to run the inference using the `*model.EstimatorServe` method.

### Training in Python using estimator and feature columns

An example of supported input is shown in the **example**: [estimator.py](test_models/estimator.py).

### Estimator serving using Go

```go
package main

import (
	"fmt"

	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/preprocessor"
	"github.com/galeone/tfgo/proto/example"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	model := tg.LoadModel("./static/1", []string{"serve"}, nil)

	// npData:numpy data like in python {"inputs":[6.4,3.2,4.5,1.5]}
	npData := make(map[string][]float32)
	npData["your_input"] = []float32{6.4, 3.2, 4.5, 1.5}
	featureExample := make(map[string]*example.Feature)
	// You need to choose the method of serialization according to your features column's type
	// e.g{"preprocessor.Float32ToFeature","preprocessor.StringToFeature","StringToFeature.Int64ToFeature"}
	featureExample["your_input"] = preprocessor.Float32ToFeature(npData["your_input"])
	seq, err := preprocessor.PythonDictToByteArray(featureExample)
	if err !=nil{
		panic(err)
	}
	newTensor, _ := tf.NewTensor([]string{string(seq)})
	results := model.EstimatorServe([]tf.Output{
		model.Op("dnn/head/predictions/probabilities", 0)}, newTensor)
	fmt.Println(results[0].Value().([][]float32))

	model = tg.LoadModel("test_models/output/2pb/", []string{"serve"}, nil)
	//pdData:pandas DataFrame like in python
	//    A    B    C    D
	//0   a    b    c    d
	//1   e    f    g    h
	// This is an example, just to show you how to deal with the features of string types.
	data := [][]string{{"a", "b", "c", "d"}, {"e", "f", "g", "h"}}
	columnsName := []string{"A", "B", "C", "D"}
	for _, item := range data {
		for index, key := range columnsName {
			featureExample[key] = preprocessor.StringToFeature([]string{item[index]})
		}
		seq, err := preprocessor.PythonDictToByteArray(featureExample)
		if err !=nil{
			panic(err)
		}
		newTensor, _ := tf.NewTensor([]string{string(seq)})
		results := model.EstimatorServe([]tf.Output{
			model.Op("dnn/head/predictions/probabilities", 0)}, newTensor)
		fmt.Println(results[0].Value().([][]float32))
	}
}

```

# Why?

Thinking about computation represented using graphs, describing computing in this way is, in one word, *challenging*.

Also, tfgo brings GPU computations to Go and allows writing parallel code without worrying about the device that executes it
(just place the graph into the device you desire: that's it!)

# Contribute

I love contributions. Seriously. Having people that share your same interests and want to face your same challenges it's something awesome.

If you'd like to contribute, just dig in the code and see what can be added or improved. Start a discussion opening an issue and let's talk about it.

Just follow the same design I use into the `image` package ("override" the same `Tensor` methods, document the methods, **test** your changes, ...)

There are **a lot** of packages that can be added, like the `image` package. Feel free to work on a brand new package: I'd love to see this kind of contributions!
