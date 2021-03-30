#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network:
    def __init__(self, args):
        hidden = inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        hidden = tf.keras.layers.Conv2D(10, 3, 2, "valid", use_bias=False)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

        hidden = tf.keras.layers.Conv2D(20, 3, 2, "valid", use_bias=False)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

        hidden = tf.keras.layers.Conv2D(20, 3, 2, "valid", use_bias=False)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)

        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = tf.keras.layers.Dense(100, activation=tf.nn.relu)(hidden)
        hidden = tf.keras.layers.Dropout(args.dropout)(hidden)

        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self._model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

    def train(self, mnist, args):
        self._model.fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        )

    def test(self, mnist, args):
        test_logs = self._model.evaluate(mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size)
        return test_logs[self._model.metrics_names.index("accuracy")]


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)
    network.test(mnist, args)
    network.model.save(os.path.join(args.logdir, "mnist_web_model.h5"), include_optimizer=False)
