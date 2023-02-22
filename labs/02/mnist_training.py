#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--decay", default=None, choices=["linear", "exponential", "cosine"], help="Decay type")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=200, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Nesterov momentum to use in SGD.")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam"], help="Optimizer to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
        tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu),
        tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
    ])

    # TODO: Use the required `args.optimizer` (either `SGD` or `Adam`).
    # - For `SGD`, if `args.momentum` is specified, use Nesterov momentum.
    # - If `args.decay` is not specified, pass the given `args.learning_rate`
    #   directly to the optimizer as a `learning_rate` argument.
    # - If `args.decay` is set, then
    #   - for `linear`, use `tf.optimizers.schedules.PolynomialDecay` with the
    #     default `power=1.0`, and set `end_learning_rate` appropriately;
    #     https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PolynomialDecay
    #   - for `exponential`, use `tf.optimizers.schedules.ExponentialDecay`,
    #     and set `decay_rate` appropriately (keep the default `staircase=False`);
    #     https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
    #   - for `cosine`, use `tf.optimizers.schedules.CosineDecay`,
    #     and set `alpha` appropriately;
    #     https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay
    #   - in all cases, you should reach the `args.learning_rate_final` just after the
    #     training, i.e., the first update after the training should use exactly the
    #     given `args.learning_rate_final`;
    #   - in all cases, `decay_steps` must be **the total number of optimizer updates**,
    #     i.e., the total number of training batches in all epochs. The size of
    #     the training MNIST dataset is `mnist.train.size`, and you can assume it
    #     is exactly divisible by `args.batch_size`.
    #   Pass the created `{Polynomial,Exponential,Cosine}Decay` to the optimizer
    #   using the `learning_rate` constructor argument.
    #
    #   If a learning rate schedule is used, TensorBoard automatically logs the last used learning
    #   rate value in every epoch. Additionally, you can find out the last used learning
    #   rate by printing `model.optimizer.learning_rate` (the original schedule is available
    #   in `model.optimizer._learning_rate` if needed), so after training, the learning rate
    #   should be close to `args.learning_rate_final` (not equal, because
    #   `model.optimizer.learning_rate` returns the last learning rate used during training).

    model.compile(
        optimizer=...,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
