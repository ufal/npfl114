#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--clip_gradient", default=None, type=float, help="Norm for gradient clipping.")
parser.add_argument("--hidden_layer", default=0, type=int, help="Additional hidden layer after RNN.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=10, type=int, help="RNN cell dimension.")
parser.add_argument("--sequence_dim", default=1, type=int, help="Sequence element dimension.")
parser.add_argument("--sequence_length", default=50, type=int, help="Sequence length.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--test_sequences", default=1000, type=int, help="Number of testing sequences.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_sequences", default=10000, type=int, help="Number of training sequences.")
# If you add more arguments, ReCodEx will keep them with your default values.


# Dataset for generating sequences, with labels predicting whether the cumulative sum
# is odd/even.
class Dataset:
    def __init__(self, sequences_num: int, sequence_length: int, sequence_dim: int, seed: int) -> None:
        sequences = np.zeros([sequences_num, sequence_length, sequence_dim], np.int32)
        labels = np.zeros([sequences_num, sequence_length, 1], np.bool8)
        generator = np.random.RandomState(seed)
        for i in range(sequences_num):
            sequences[i, :, 0] = generator.randint(0, max(2, sequence_dim), size=[sequence_length])
            labels[i, :, 0] = np.bitwise_and(np.cumsum(sequences[i, :, 0]), 1)
            if sequence_dim > 1:
                sequences[i] = np.eye(sequence_dim)[sequences[i, :, 0]]
        self._data = {"sequences": sequences.astype(np.float32), "labels": labels}
        self._size = sequences_num

    @property
    def data(self) -> Dict[str, np.ndarray]:
        return self._data

    @property
    def size(self) -> int:
        return self._size


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # Construct the model.
        sequences = tf.keras.layers.Input(shape=[args.sequence_length, args.sequence_dim])

        # TODO: Process the sequence using a RNN with cell type `args.rnn_cell`
        # and with dimensionality `args.rnn_cell_dim`. Use `return_sequences=True`
        # to get outputs for all sequence elements.
        #
        # Prefer `tf.keras.layers.{LSTM,GRU,SimpleRNN}` to
        # `tf.keras.layers.RNN` wrapper with `tf.keras.layers.{LSTM,GRU,SimpleRNN}Cell`,
        # because the former can run transparently on a GPU and is also
        # considerably faster on a CPU).

        # TODO: If `args.hidden_layer` is nonzero, process the result using
        # a ReLU-activated fully connected layer with `args.hidden_layer` units.

        # TODO: Generate `predictions` using a fully connected layer
        # with one output and `tf.nn.sigmoid` activation.

        super().__init__(inputs=sequences, outputs=predictions)

        self.compile(
            optimizer=tf.optimizers.Adam(clipnorm=args.clip_gradient),
            loss=tf.losses.BinaryCrossentropy(),
            metrics=[tf.metrics.BinaryAccuracy("accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


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

    # Create the data
    train = Dataset(args.train_sequences, args.sequence_length, args.sequence_dim, seed=42)
    test = Dataset(args.test_sequences, args.sequence_length, args.sequence_dim, seed=43)

    # Create the model and train
    model = Model(args)
    logs = model.fit(
        train.data["sequences"], train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(test.data["sequences"], test.data["labels"]),
        callbacks=[model.tb_callback],
    )

    # Return development metrics for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
