# tfgo: Tensorflow in Go

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
root := tfgo.NewRoot()
```

We can now place nodes into this graphs and connect them. Let's say we want to multiply a matrix for a column vector and then add another column vector to the result.

Here's the complete source code.

```go
package main

import (
        "fmt"
        "github.com/galeone/tfgo"
        tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
        root := tfgo.NewRoot()
        A := tfgo.NewTensor(root, tfgo.Const(root, [2][2]int32{{1, 2}, {-1, -2}}))
        x := tfgo.NewTensor(root, tfgo.Const(root, [2][1]int64{{10}, {100}}))
        b := tfgo.NewTensor(root, tfgo.Const(root, [2][1]int32{{-10}, {10}}))
        Y := A.MatMul(x.Output).Add(b.Output)
        // Please note that Y is just a pointer to A!

        // If we want to create a different node in the graph, we have to clone Y
        // or equivalently A
        Z := A.Clone()
        results := tfgo.Exec(root, []tf.Output{Y.Output, Z.Output}, nil, &tf.SessionOptions{})
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

Tensorflow is rich of methods for performing operations on images. tfgo provides the `image` package that allows using the Go bindings to perform computer vision tasks in a elegant way.

For instance, it's possible to read an image, compute its directional derivative along the horizontal and vertical directions, compute the gradient and save it.

The code below does that, showing the different results achieved using correlation and convolution operations.

```go
package main

import (
        "github.com/galeone/tfgo"
        "github.com/galeone/tfgo/image"
        "github.com/galeone/tfgo/image/filter"
        "github.com/galeone/tfgo/image/padding"
        tf "github.com/tensorflow/tensorflow/tensorflow/go"
        "os"
)

func main() {
        root := tfgo.NewRoot()
        grayImg := image.Read(root, "/home/pgaleone/test_sobel.PNG", 1)
        grayImg = grayImg.Scale(0, 255)

        // Edge detection using sobel filter: convolution
        Gx := grayImg.Convolve(filter.SobelX(root), image.Stride{X: 1, Y: 1}, padding.SAME).Clone()
        Gy := grayImg.Convolve(filter.SobelY(root), image.Stride{X: 1, Y: 1}, padding.SAME).Clone()
        convoluteEdges := image.NewImage(root.SubScope("edge"), Gx.Square().Add(Gy.Square().Value()).Sqrt().Value()).EncodeJPEG()

        Gx = grayImg.Correlate(filter.SobelX(root), image.Stride{X: 1, Y: 1}, padding.SAME).Clone()
        Gy = grayImg.Correlate(filter.SobelY(root), image.Stride{X: 1, Y: 1}, padding.SAME).Clone()
        correlateEdges := image.NewImage(root.SubScope("edge"), Gx.Square().Add(Gy.Square().Value()).Sqrt().Value()).EncodeJPEG()

        results := tfgo.Exec(root, []tf.Output{convoluteEdges, correlateEdges}, nil, &tf.SessionOptions{})
        file, _ := os.Create("convolve.png")
        file.WriteString(results[0].Value().(string))
        file.Close()

        file, _ = os.Create("correlated.png")
        file.WriteString(results[1].Value().(string))
        file.Close()
}

```

**airplane.png**

![airplane](https://i.imgur.com/QS6shgc.jpg)

**correlated.jpg**

![correlated](https://i.imgur.com/vhYF7o3.jpg)

**convolved.jpg**

![convolved](https://i.imgur.com/zVndo9B.jpg)

the list of the available methods is available on GoDoc: http://godoc.org/github.com/galeone/tfgo/image

# Why?

Thinking about computation represented using graphs, describing computing in this way is, in one word, *challenging*.

Also, tfgo brings GPU computations to Go and allows writing parallel code without worrying about the device that executes it
(just place the graph into the device you desire: that's it!)

# Contribute

I love contributions. Seriously. Having people that share your same interests and want to face your same challenges it's something awsome.

If you'd like to contribute, just dig in the code and see what can be added or improved. Start a dicussion opening an issue and let's talk about it.

Just follow the same design I use into the `image` package ("override" the same `Tensor` methods, document the methods, **test** your changes, ...)

There are **a lot** of packages that can be added, like the `image` package. Feel free to work on a brand new package: I'd love to see this kind of contribution!
