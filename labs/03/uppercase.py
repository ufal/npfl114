#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, notably
# for `alphabet_size` and `window` and others.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=None, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=None, type=int, help="Window size to use.")

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
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters and train the model.
    #
    # The inputs are _windows_ of fixed size (`args.window` characters on left,
    # the character in question, and `args.window` characters on right), where
    # each character is represented by a `tf.int32` index. To suitably represent
    # the characters, you can:
    # - Convert the character indices into _one-hot encoding_. There is no
    #   explicit Keras layer, but you can
    #   - use a Lambda layer which can encompass any function:
    #       tf.keras.Sequential([
    #         tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
    #         tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
    #   - or use Functional API and then any TF function can be used
    #     as a Keras layer:
    #       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
    #       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
    #   You can then flatten the one-hot encoded windows and follow with a dense layer.
    # - Alternatively, you can use `tf.keras.layers.Embedding` (which is an efficient
    #   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.
    model = ...

    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to predictions_file (which is
    # `uppercase_test.txt` in the `args.logdir` directory).
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        ...

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
