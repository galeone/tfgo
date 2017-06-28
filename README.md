# tfgo: Tensorflow in Go

Tensorflow's Go bindings are [hard to use](https://pgaleone.eu/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/): tfgo makes it easy!

No more problems like:

- Scoping: each new node will have a new and unique name
- Typing: attributes are automatically converted to a supported type instead of throwing errors at runtime

Also, it uses [Method chaining](https://en.wikipedia.org/wiki/Method_chaining) making possible to write pleasant Go code.

# Getting started

The core data structure of the Tensorflow's Go bindings is the `op.Scope` struct. `tfgo` allows creating new `*op.Scope` that solve the scoping issue mentioned above.

Since we're defining a graph, let's start from its root

```go
root := tfgo.NewRoot()
```


# Why?

It's challenging.

Also, tfgo brings GPU computations to Go and allows writing parallel code without worring about the device that execute it
(just place the graph into the device you desire: that's it!)
