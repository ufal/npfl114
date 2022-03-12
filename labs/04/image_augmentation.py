#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    cifar = CIFAR10(size={"train": 5000, "dev": 1000})

    # Create the model
    inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
    hidden = tf.keras.layers.Conv2D(16, 3, 2, "same", activation=tf.nn.relu)(inputs)
    hidden = tf.keras.layers.Conv2D(16, 3, 1, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(24, 3, 2, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(24, 3, 1, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(32, 3, 2, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(32, 3, 1, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(200, activation=tf.nn.relu)(hidden)
    outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(hidden)

    # Train the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    # TODO: Create data augmenting `tf.keras.preprocessing.image.ImageDataGenerator`.
    # Specify:
    # - rotation range of 20 degrees,
    # - zoom range of 0.2 (20%),
    # - width shift range and height shift range of 0.1 (10%),
    # - allow horizontal flips
    train_generator = ...

    # TODO: Train using the generator. To augment data, use
    # `train_generator.flow` and specify:
    # - `cifar.train.data["images"]` as inputs
    # - `cifar.train.data["labels"]` as target
    # - batch_size of `args.batch_size`
    # - `args.seed` as the random seed
    logs = model.fit(
        ...,
        shuffle=False, epochs=args.epochs,
        validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    # Return development metrics for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
