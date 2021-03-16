#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[100], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

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

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    for epoch in range(args.epochs):
        model.reset_metrics()
        for batch in mnist.train.batches(args.batch_size):
            train = model.train_on_batch(batch["images"], batch["labels"], reset_metrics=False, return_dict=True)

        model.reset_metrics()
        for batch in mnist.dev.batches(args.batch_size):
            dev = model.test_on_batch(batch["images"], batch["labels"], reset_metrics=False, return_dict=True)
        print("Epoch {} finished, train: {}, dev: {}".format(epoch + 1, train, dev))

    model.reset_metrics()
    for batch in mnist.test.batches(args.batch_size):
        test = model.test_on_batch(batch["images"], batch["labels"], reset_metrics=False, return_dict=True)
    print("Test: {}".format(test))

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
