#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

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
    def __init__(self, sequences_num, sequence_length, sequence_dim, seed, shuffle_batches=True):
        sequences = np.zeros([sequences_num, sequence_length, sequence_dim], np.int32)
        labels = np.zeros([sequences_num, sequence_length, 1], np.bool)
        generator = np.random.RandomState(seed)
        for i in range(sequences_num):
            sequences[i, :, 0] = generator.randint(0, max(2, sequence_dim), size=[sequence_length])
            labels[i, :, 0] = np.bitwise_and(np.cumsum(sequences[i, :, 0]), 1)
            if sequence_dim > 1:
                sequences[i] = np.eye(sequence_dim)[sequences[i, :, 0]]
        self._data = {"sequences": sequences.astype(np.float32), "labels": labels}
        self._size = sequences_num

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self._size

class Network(tf.keras.Model):
    def __init__(self, args):
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
            optimizer=tf.optimizers.Adam(global_clipnorm=args.clip_gradient),
            loss=tf.losses.BinaryCrossentropy(),
            metrics=[tf.metrics.BinaryAccuracy("accuracy")],
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, write_graph=False, profile_batch=0)

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

    # Create the data
    train = Dataset(args.train_sequences, args.sequence_length, args.sequence_dim, seed=42, shuffle_batches=True)
    test = Dataset(args.test_sequences, args.sequence_length, args.sequence_dim, seed=43, shuffle_batches=False)

    # Create the network and train
    network = Network(args)
    logs = network.fit(
        train.data["sequences"], train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(test.data["sequences"], test.data["labels"]),
        callbacks=[network.tb_callback],
    )

    # Return test set accuracy for ReCodEx to validate
    return logs.history["val_accuracy"][-1]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
