#Copyright (C) 2017 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Use DyTB to define and train a model. Then redefines it, changing the input
while restoring the learned weights of the best model. Then export it in a protobuf."""

import sys
import tensorflow as tf
from dytb.inputs.predefined.MNIST import MNIST
from dytb.models.predefined.LeNetDropout import LeNetDropout
from dytb.train import train


def main():
    """main executes the operations described in the module docstring"""
    lenet = LeNetDropout()
    mnist = MNIST()

    info = train(model=lenet, dataset=mnist, hyperparameters={"epochs": 1})

    checkpoint_path = info["paths"]["best"]

    with tf.Session() as sess:
        # Define a new model, import the weights from best model trained
        # Change the input structure to use a placeholder
        images = tf.placeholder(
            tf.float32, shape=(None, 28, 28, 1), name="input_")
        # define in the default graph the model that uses placeholder as input
        _ = lenet.get(images, mnist.num_classes)

        # The best checkpoint path contains just one checkpoint, thus the last is the best
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

        # Create a builder to export the model
        builder = tf.saved_model.builder.SavedModelBuilder("export")
        # Tag the model in order to be capable of restoring it specifying the tag set
        builder.add_meta_graph_and_variables(sess, ["tag"])
        builder.save()

    return 0


if __name__ == '__main__':
    sys.exit(main())
