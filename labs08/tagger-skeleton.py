#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

import morpho_dataset

class Network:
    def __init__(self, rnn_cell, rnn_cell_dim, method, words, logdir, expname, threads=1, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.train.SummaryWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)

        # Construct the graph
        with self.session.graph.as_default():
            if rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
            elif rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
            else:
                raise ValueError("Unknown rnn_cell {}".format(rnn_cell))

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.forms = tf.placeholder(tf.int32, [None, None])
            self.tags = tf.placeholder(tf.int32, [None, None])

            # TODO
            # loss = ...
            # self.training = ...
            # self.predictions = ...
            # self.accuracy = ...

            self.dataset_name = tf.placeholder(tf.string, [])
            self.summary = tf.merge_summary([tf.scalar_summary(self.dataset_name+"/loss", loss),
                                             tf.scalar_summary(self.dataset_name+"/accuracy", self.accuracy)])

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, sentence_lens, forms, tags):
        _, summary = self.session.run([self.training, self.summary],
                                      {self.sentence_lens: sentence_lens, self.forms: forms,
                                       self.tags: tags, self.dataset_name: "train"})
        self.summary_writer.add_summary(summary, self.training_step)

    def evaluate(self, sentence_lens, forms, tags):
        accuracy, summary = self.session.run([self.accuracy, self.summary],
                                   {self.sentence_lens: sentence_lens, self.forms: forms,
                                    self.tags: tags, self.dataset_name: "dev"})
        self.summary_writer.add_summary(summary, self.training_step)
        return accuracy

    def predict(self, sentence_lens, forms):
        return self.session.run(self.predictions,
                                {self.sentence_lens: sentence_lens, self.forms: forms})


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--data_train", default="en-train.txt", type=str, help="Training data file.")
    parser.add_argument("--data_dev", default="en-dev.txt", type=str, help="Development data file.")
    parser.add_argument("--data_test", default="en-test.txt", type=str, help="Testing data file.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--method", default="learned_we", type=str, help="Which method of word embeddings to use.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=100, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    print("Loading the data.", file=sys.stderr)
    data_train = morpho_dataset.MorphoDataset(args.data_train, add_bow_eow=True)
    data_dev = morpho_dataset.MorphoDataset(args.data_dev, train=data_train)
    data_test = morpho_dataset.MorphoDataset(args.data_test, train=data_train)

    # Construct the network
    print("Constructing the network.", file=sys.stderr)
    expname = "tagger-{}{}-m{}-bs{}-epochs{}".format(args.rnn_cell, args.rnn_cell_dim, args.method, args.batch_size, args.epochs)
    network = Network(rnn_cell=args.rnn_cell, rnn_cell_dim=args.rnn_cell_dim, method=args.method,
                      words=len(data_train.factors[data_train.FORMS]['words']),
                      logdir=args.logdir, expname=expname, threads=args.threads)

    # Train
    best_dev_accuracy = 0
    test_predictions = None

    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch + 1), file=sys.stderr)
        while not data_train.epoch_finished():
            sentence_lens, word_ids = data_train.next_batch(args.batch_size)
            network.train(sentence_lens, word_ids[data_train.FORMS], word_ids[data_train.TAGS])
            # To use character-level embeddings, pass including_charseqs=True to next_batch
            # and instead of word_ids[data_train.FORMS] use charseq_ids[data_train.FORMS],
            # charseqs[data_train.FORMS] and charseq_lens[data_train.FORMS]

        dev_sentence_lens, dev_word_ids = data_dev.whole_data_as_batch()
        dev_accuracy = network.evaluate(dev_sentence_lens, dev_word_ids[data_dev.FORMS], dev_word_ids[data_dev.TAGS])
        print("Development accuracy after epoch {} is {:.2f}.".format(epoch + 1, 100. * dev_accuracy), file=sys.stderr)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            test_sentence_lens, test_word_ids = data_test.whole_data_as_batch()
            test_predictions = network.predict(test_sentence_lens, test_word_ids[data_test.FORMS])

    # Print test predictions
    test_forms = data_test.factors[data_test.FORMS]['strings'] # We use strings instead of words, because words can be <unk>
    test_tags = data_test.factors[data_test.TAGS]['words']
    for i in range(len(data_test.sentence_lens)):
        for j in range(data_test.sentence_lens[i]):
            print("{}\t_\t{}".format(test_forms[i][j], test_tags[test_predictions[i, j]]))
        print()
