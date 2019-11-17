# Protobuf files

These files have been kept from the TensorFlow repository.

The goal of having them here, is to be able to correctly serialize data in the correct proto-message and thus create correctly a `*tf.Tensor` object, when the input to serialize of a "complex" data type.

Complex, in this case, doesn't mean complex number, but not a basic type (e.g. not a float, or a float array, but a Python dictionary that has been used to train a model).

### Copyright notice

All rights are reserved to the TensorFlow team - the only modification made is the package name in the `.proto` files.
