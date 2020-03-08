#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layers", default="100", type=str, help="Hidden layer sizes separated by comma.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the model
    inputs = tf.keras.layers.Input([MNIST.H, MNIST.W, MNIST.C])
    hidden = tf.keras.layers.Flatten()(inputs)
    for hidden_layer in args.hidden_layers:
        hidden = tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu)(hidden)
    outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(args.epochs):
        accuracy.reset_states()
        for batch in mnist.train.batches(args.batch_size):
            with tf.GradientTape() as tape:
                probabilities = model(batch["images"], training=True)
                loss = loss_fn(batch["labels"], probabilities)
                accuracy(batch["labels"], probabilities)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train = accuracy.result()

        accuracy.reset_states()
        for batch in mnist.dev.batches(args.batch_size):
            probabilities = model(batch["images"], training=False)
            accuracy(batch["labels"], probabilities)
        dev = accuracy.result()
        print("Epoch {} finished, train: {}, dev: {}".format(epoch + 1, train, dev))

    accuracy.reset_states()
    for batch in mnist.test.batches(args.batch_size):
        probabilities = model(batch["images"], training=False)
        accuracy(batch["labels"], probabilities)
    test = accuracy.result()
    print("Test: {}".format(test))
