#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset

class Network:
    def __init__(self, args, num_words, num_tags, num_chars):
        # TODO(we): Implement a one-layer RNN network. The input
        # `word_ids` consists of a batch of sentences, each
        # a sequence of word indices. Padded words have index 0.

        # TODO: Apart from `word_ids`, RNN CLEs utilize two more
        # inputs, `charseqs` containing unique words in batches (each word
        # being a sequence of character indices, padding characters again
        # have index 0) and `charseq_ids` with the same shape as `word_ids`,
        # but with indices pointing into `charseqs`.

        # TODO: Embed the characters in `charseqs` using embeddings of size
        # `args.cle_dim`, masking zero indices. Then, pass the embedded characters
        # through a bidirectional GRU with dimension `args.cle_dim`, concatenating
        # results in different dimensions.
        #
        # Then, copy the computed embeddings of unique words to the correct sentence
        # positions. To that end, use `tf.gather` operation, which is given a matrix
        # and a tensor of indices, and replace each index by a corresponding row
        # of the matrix. You need to wrap the `tf.gather` in `tf.keras.layers.Lambda`
        # because of a bug [fixed 6 days ago in the master], so the call shoud look like
        # `tf.keras.layers.Lambda(lambda args: tf.gather(*args))(...)`

        # TODO(we): Embed input words with dimensionality `args.we_dim`, using
        # `mask_zero=True`.

        # TODO: Concatenate the WE and CLE embeddings (in this order).

        # TODO(we): Create specified `args.rnn_cell` rnn cell (LSTM, GRU) with
        # dimension `args.rnn_cell_dim` and apply it in a bidirectional way on
        # the embedded words, concatenating opposite directions.

        # TODO(we): Add a softmax classification layer into `num_tags` classes, storing
        # the outputs in `predictions`.

        self.model = tf.keras.Model(inputs=[word_ids, charseq_ids, charseqs], outputs=predictions)

        # TODO: Create an Adam optimizer in self._optimizer
        # TODO: Create a suitable loss in self._loss
        # TODO: Create two metrics in self._metrics dictionary:
        #  - "loss", which is tf.metrics.Mean()
        #  - "accuracy", which is suitable accuracy
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3,
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def train_batch(self, inputs, tags):
        # TODO: Generate a mask from `tags` containing ones in positions
        # where tags are nonzero (using `tf.not_equal`).
        with tf.GradientTape() as tape:
            probabilities = self.model(inputs, training=True)
            # TODO: Compute `loss` using `self._loss`, passing the generated
            # tag mask as third parameter.
        gradients = tape.gradient(loss, self.model.variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.variables))

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss": metric(loss)
                else: # TODO: Update the `metric` using gold `tags` and generated `probabilities`,
                      # passing the tag mask as third argument.
                tf.summary.scalar("train/{}".format(name), metric.result())

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            self.train_batch([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs],
                             batch[dataset.TAGS].word_ids)

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3,
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def evaluate_batch(self, inputs, tags):
        # TODO: Again generate a mask from `tags` containing ones in positions
        # where tags are nonzero (using `tf.not_equal`).
        probabilities = self.model(inputs, training=False)
        # TODO: Compute `loss` using `self._loss`, passing the generated
        # tag mask as third parameter.
        for name, metric in self._metrics.items():
            if name == "loss": metric(loss)
            else: # TODO: Update the `metric` using gold `tags` and generated `probabilities`,
                  # passing the tag mask as third argument.

    def evaluate(self, dataset, dataset_name, args):
        for metric in self._metrics.values():
            metric.reset_states()
        for batch in dataset.batches(args.batch_size):
            # TODO: Evaluate the given match, using the same inputs as in training.

        metrics = {name: metric.result() for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, name), value)

        return metrics


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--max_sentences", default=5000, type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.initializers.glorot_uniform(seed=42)
        tf.keras.utils.get_custom_objects()["orthogonal"] = lambda: tf.initializers.orthogonal(seed=42)
        tf.keras.utils.get_custom_objects()["uniform"] = lambda: tf.initializers.RandomUniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the network and train
    network = Network(args,
                      num_words=len(morpho.train.data[morpho.train.FORMS].words),
                      num_tags=len(morpho.train.data[morpho.train.TAGS].words),
                      num_chars=len(morpho.train.data[morpho.train.FORMS].alphabet))
    for epoch in range(args.epochs):
        network.train_epoch(morpho.train, args)
        metrics = network.evaluate(morpho.dev, "dev", args)

    metrics = network.evaluate(morpho.test, "test", args)
    with open("tagger_we.out", "w") as out_file:
        print("{:.2f}".format(100 * metrics["accuracy"]), file=out_file)
