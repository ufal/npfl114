#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0, type=float, help="Dropout regularization.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[400], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--l2", default=0, type=float, help="L2 regularization.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # TODO: Create the model and incorporate the L2 regularization and dropout:
    # - L2 regularization:
    #   If `args.l2` is nonzero, create a `tf.keras.regularizers.L1L2` regularizer
    #   and use it for all kernels (but not biases) of all Dense layers.
    # - Dropout:
    #   Add a `tf.keras.layers.Dropout` with `args.dropout` rate after the Flatten
    #   layer and after each Dense hidden layer (but not after the output Dense layer).

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]))
    for hidden_layer in args.hidden_layers:
        model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax))

    # TODO: Implement label smoothing.
    # Apply the given smoothing. You will need to change the
    # `SparseCategorical{Crossentropy,Accuracy}` to `Categorical{Crossentropy,Accuracy}`
    # because `label_smoothing` is supported only by `CategoricalCrossentropy`.
    # That means you also need to modify the labels of all three datasets
    # (i.e., `mnist.{train,dev,test}.data["labels"]`) from indices of the gold class
    # to a full categorical distribution (you can use either NumPy or there is
    # a helper method also in the `tf.keras.utils`).

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    tb_callback._close_writers = lambda: None # Ugly hack allowing to log also test data metrics.
    model.fit(
        mnist.train.data["images"][:5000], mnist.train.data["labels"][:5000],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    test_logs = model.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size, return_dict=True,
    )
    tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    return test_logs["accuracy"]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
