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

# Computer Vision using data flow graph

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

# Train in Python, Serve in Go

Using both [DyTB](https://github.com/galeone/dynamic-training-bench) and tfgo we can train, evaluate and export a machine learning model in very few lines of Python and Go code. Below you can find the Python and the Go code.
Just dig into the example to understand how to serve a trained model with `tfgo`.

**Python code**:

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

**Go code**:

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

# Train by python with tf.estimator,Serve in go
**python train code**
```
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class EStrain(object):

    def __init__(self):
        self.iris = load_iris()

    def get_train_test(self):
        data = self.iris.data
        target = self.iris.target
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def get_feature_columns_by_numpy(self):
        columns = [
            tf.feature_column.numeric_column("your_input", shape=(4,))
        ]
        return columns

    def get_feature_columns_by_pandas(self):
        columns = [
            tf.feature_column.numeric_column(name,shape=(1,)) for name in list("abcd")
        ]
        return columns

    def input_fn_by_numpy(self, x, y):
        return tf.estimator.inputs.numpy_input_fn(
            x={"your_input": x},
            y=y,
            batch_size=512,
            num_epochs=1,
            shuffle=False,
            queue_capacity=1000,
            num_threads=1
        )

    def input_fn_by_pandas(self, x, y):
        return tf.estimator.inputs.pandas_input_fn(
            x,
            y,
            batch_size=32,
            num_epochs=1,
            shuffle=False,
            queue_capacity=1000,
            num_threads=1
        )

    def to_pandas(self, arr, columns):
        return pd.DataFrame(arr, columns=columns)

    def get_est(self, path, feature_columns):
        est = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[10, 20, 10],
            n_classes=3,
            model_dir=path
        )
        return est

    def train_by_numpy(self):
        x_train, x_test, y_train, y_test = self.get_train_test()
        feature_columns = self.get_feature_columns_by_numpy()
        est = self.get_est("./output/1", feature_columns)
        train_input = self.input_fn_by_numpy(x_train, y_train)
        test_input = self.input_fn_by_numpy(x_test, y_test)
        est.train(input_fn=train_input)
        accuracy_score = est.evaluate(input_fn=test_input)["accuracy"]
        print("accuracy:%s\n" % accuracy_score)
        """ a test example"""
        samples = np.array([[6.4, 3.2, 4.5, 1.5],
                            [6.4, 3.2, 4.5, 1.5]
                            ])
        samples_input = self.input_fn_by_numpy(samples, None)
        predictions = list(est.predict(samples_input))
        print(predictions)
        predicted_classes = int(predictions[0]["classes"])
        print("predict result is %s\n" % predicted_classes)

    def train_by_pandas(self):
        x_train, x_test, y_train, y_test = self.get_train_test()
        feature_columns = self.get_feature_columns_by_pandas()
        est = self.get_est("./output/2", feature_columns)
        x_train_pd = self.to_pandas(x_train, columns=list("abcd"))
        x_test_pd = self.to_pandas(x_test, columns=list("abcd"))
        y_train_pd = pd.Series(y_train)
        y_test_pd = pd.Series(y_test)
        train_input = self.input_fn_by_pandas(x_train_pd, y_train_pd)
        test_input = self.input_fn_by_pandas(x_test_pd, y_test_pd)
        est.train(input_fn=train_input)
        accuracy_score = est.evaluate(input_fn=test_input)["accuracy"]
        print("accuracy:%s\n" % accuracy_score)
        """ a test example"""
        samples = pd.DataFrame(
            [[6.4, 3.2, 4.5, 1.5]], columns=list("abcd")
        )
        samples_input = self.input_fn_by_pandas(samples, None)
        predictions = list(est.predict(samples_input))
        print(predictions)
        predicted_classes = int(predictions[0]["classes"])
        print("predict result is %s\n" % predicted_classes)


if __name__ == '__main__':
    et = EStrain()
    et.train_by_pandas()
    et.train_by_numpy()
```
**ckpt to pb**

```python
from train import EStrain
import tensorflow as tf


class ConvertToPB(object):

    def __init__(self):
        self.model_dir_np = "./output/1"
        self.model_dir_pd = "./output/2"

    def serving_input_receiver_fn(self, feature_spec):
        serizlized_ft_example = tf.placeholder(dtype=tf.float64, shape=[None, 4], name="input_tensor")
        receiver_tensors = {"input": serizlized_ft_example}
        features = tf.parse_example(serizlized_ft_example, feature_spec)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    def convert_np(self):
        es = EStrain()
        feature_columns = es.get_feature_columns_by_numpy()
        est = es.get_est(self.model_dir_np, feature_columns)
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        est.export_saved_model("./output/1pb", export_input_fn, as_text=True)

    def convert_pd(self):
        es = EStrain()
        feature_columns = es.get_feature_columns_by_pandas()
        est = es.get_est(self.model_dir_pd, feature_columns)
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        est.export_saved_model("./output/2pb", export_input_fn, as_text=True)


if __name__ == '__main__':
    ct = ConvertToPB()
    ct.convert_np()
    ct.convert_pd()

```
**Serve by go**
```go
package main

import (
	"fmt"
	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/core/example"
	"github.com/galeone/tfgo/train"
	"github.com/gogo/protobuf/proto"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	data := [][]float32{{6.4, 3.2, 4.5, 1.5}, {100., 34.5, 4.5, 3.5}}
	columnsName := []string{"a", "b", "c", "d"}
	for _, item := range data {
		// npData:numpy data like in python {"inputs":[6.4,3.2,4.5,1.5]}
		npData := make(map[string][]float32)
		npData["your_input"] = item
		predictNP(npData)
		// pdData:pandas DataFrame like in python
		//     a    b    c    d
		// 0  6.4  3.4  4.5  1.5
		pdData := make(map[string]float32)
		for index, key := range columnsName {
			pdData[key] = item[index]
		}
		predictPD(pdData)
	}

}

func loadModeSavedPB(path string) (model *tg.Model) {
	model = tg.LoadModel(path, []string{"serve"}, nil)
	return
}

func predictNP(data map[string][]float32) {
	npModePath := "./static/1"
	model := loadModeSavedPB(npModePath)
	sequence, err := sequenceNP(data)
	if err != nil {
		panic(err)
	}
	fakeInput, _ := tf.NewTensor([]string{string(sequence)})
	results := model.Exec([]tf.Output{
		model.Op("dnn/head/predictions/probabilities", 0),
	}, map[tf.Output]*tf.Tensor{
		model.Op("input_example_tensor", 0): fakeInput,
	})
	predictions := results[0].Value().([][]float32)
	fmt.Println(predictions)
}

func predictPD(data map[string]float32) {
	pdModelPath := "./static/2"
	model := loadModeSavedPB(pdModelPath)
	sequence, err := sequencePD(data)
	if err != nil {
		panic(err)
	}
	fakeInput, _ := tf.NewTensor([]string{string(sequence)})
	results := model.Exec([]tf.Output{
		model.Op("dnn/head/predictions/probabilities", 0),
	}, map[tf.Output]*tf.Tensor{
		model.Op("input_example_tensor", 0): fakeInput,
	})
	predictions := results[0].Value().([][]float32)
	fmt.Println(predictions)
}

func sequenceNP(featureInfo map[string][]float32) (seq []byte, err error) {
	feature := make(map[string]*example.Feature)
	for k, v := range featureInfo {
		valFormat := train.Float32ToFeature(v)
		feature[k] = valFormat
	}
	Features := example.Features{Feature: feature}
	myExample := example.Example{Features: &Features}
	seq, err = proto.Marshal(&myExample)
	return
}

func sequencePD(featureInfo map[string]float32) (seq []byte, err error) {
	feature := make(map[string]*example.Feature)
	for k, v := range featureInfo {
		valFormat := train.Float32ToFeature([]float32{v})
		feature[k] = valFormat
	}
	Features := example.Features{Feature: feature}
	myExample := example.Example{Features: &Features}
	seq, err = proto.Marshal(&myExample)
	return
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
