#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, sequences, sequence_length, sequence_dim, shuffle_batches=True):
        self._sequences = np.zeros([sequences, sequence_length, sequence_dim], np.int32)
        self._labels = np.zeros([sequences, sequence_length], np.bool)

        for i in range(sequences):
            self._sequences[i, :, 0] = np.random.random_integers(0, max(1, sequence_dim - 1), size=[sequence_length])
            self._labels[i] = np.bitwise_and(np.cumsum(self._sequences[i, :, 0]), 1)
            if sequence_dim > 1:
                self._sequences[i] = np.eye(sequence_dim)[self._sequences[i, :, 0]]

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._sequences)) if self._shuffle_batches else np.arange(len(self._sequences))

    @property
    def sequences(self):
        return self._sequences

    @property
    def labels(self):
        return self._labels

    def all_data(self):
        return self._sequences, self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._sequences[batch_perm], self._labels[batch_perm]

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sequences)) if self._shuffle_batches else np.arange(len(self._sequences))
            return True
        return False


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.sequences = tf.placeholder(tf.float32, [None, args.sequence_length, args.sequence_dim], name="sequences")
            self.labels = tf.placeholder(tf.bool, [None, args.sequence_length], name="labels")

            # TODO: Create RNN cell according to args.rnn_cell (RNN, LSTM and GRU should be supported,
            # using BasicRNNCell, BasicLSTMCell and GRUCell from tf.nn.rnn_cell module),
            # with dimensionality of args.rnn_cell_dim. Store the cell in `rnn_cell`.

            # TODO: Process self.sequences using `tf.nn.dynamic_rnn` and `rnn_cell`,
            # store the outputs to `hidden_layer` and ignore output states.

            # TODO: If args.hidden_layer, add a dense layer with `args.hidden_layer` neurons
            # and ReLU activation.

            # TODO: Add a dense layer with one output neuron, without activation, into `output_layer`

            # TODO: Remove the third dimension from `output_layer` using `tf.squeeze`.

            # TODO: Generate self.predictions with either False/True according to whether
            # values in `output_layer` are less or grater than 0 (using `tf.greater_equal`).
            # This corresponds to rounding the probability of sigmoid applied to `output_layer`.

            # Training
            loss = tf.losses.sigmoid_cross_entropy(tf.cast(self.labels, tf.int32), output_layer)
            global_step = tf.train.create_global_step()
            optimizer = tf.train.AdamOptimizer()
            # Note how instead of `optimizer.minimize` we first get the # gradients using
            # `optimizer.compute_gradients`, then optionally clip them and
            # finally apply then using `optimizer.apply_gradients`.
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            # TODO: Compute norm of gradients using `tf.global_norm` into `gradient_norm`.
            # TODO: If args.clip_gradient, clip gradients (back into `gradients`) using `tf.clip_by_global_norm`.
            self.training = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/gradient_norm", gradient_norm),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, sequences, labels):
        self.session.run([self.training, self.summaries["train"]], {self.sequences: sequences, self.labels: labels})

    def evaluate(self, dataset, sequences, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.sequences: sequences, self.labels: labels})
        return accuracy


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--clip_gradient", default=None, type=float, help="Norm for gradient clipping.")
    parser.add_argument("--hidden_layer", default=None, type=int, help="Additional hidden layer after RNN.")
    parser.add_argument("--dev_sequences", default=1000, type=int, help="Number of development sequences.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=10, type=int, help="RNN cell dimension.")
    parser.add_argument("--sequence_dim", default=1, type=int, help="Sequence element dimension.")
    parser.add_argument("--sequence_length", default=50, type=int, help="Sequence length.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--train_sequences", default=10000, type=int, help="Number of training sequences.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset(args.train_sequences, args.sequence_length, args.sequence_dim)
    dev = Dataset(args.dev_sequences, args.sequence_length, args.sequence_dim, shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            sequences, labels = train.next_batch(args.batch_size)
            network.train(sequences, labels)

        dev_sequences, dev_labels = dev.all_data()
        print("{:.2f}".format(100 * network.evaluate("dev", dev_sequences, dev_labels)))
