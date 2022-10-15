# Copyright (C) 2017-2022 Paolo Galeone <nessuno@nerdz.eu>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# Exhibit B is not attached; this software is compatible with the
# licenses expressed under Section 1.12 of the MPL v2.

import sys

import tensorflow as tf


def keras():
    """Define a trivial module for image (28x28x1) classification.
    Export it as a SavedModel without even training it.

    Rawly serialize an uninitialized Keras Sequential model."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                8,
                (3, 3),
                strides=(2, 2),
                padding="valid",
                input_shape=(28, 28, 1),
                activation=tf.nn.relu,
                name="inputs",
            ),  # 14x14x8
            tf.keras.layers.Conv2D(
                16, (3, 3), strides=(2, 2), padding="valid", activation=tf.nn.relu
            ),  # 7x716
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, name="logits"),  # linear
        ]
    )

    tf.saved_model.save(model, "output/keras")


def tf_function():
    pass


def main():
    tf.io.gfile.makedirs("output")

    keras()
    tf_function()


if __name__ == "__main__":
    sys.exit(main())
