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
    def __init__(self, rnn_cell, rnn_cell_dim, num_chars, bow_char, eow_char, logdir, expname, threads=1, seed=42):
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
            self.form_ids = tf.placeholder(tf.int32, [None, None])
            self.forms = tf.placeholder(tf.int32, [None, None])
            self.form_lens = tf.placeholder(tf.int32, [None])
            self.lemma_ids = tf.placeholder(tf.int32, [None, None])
            self.lemmas = tf.placeholder(tf.int32, [None, None])
            self.lemma_lens = tf.placeholder(tf.int32, [None])

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

    def train(self, sentence_lens, form_ids, forms, form_lens, lemma_ids, lemmas, lemma_lens):
        _, summary = self.session.run([self.training, self.summary],
                                      {self.sentence_lens: sentence_lens,
                                       self.form_ids: form_ids, self.forms: forms, self.form_lens: form_lens,
                                       self.lemma_ids: lemma_ids, self.lemmas: lemmas, self.lemma_lens: lemma_lens,
                                       self.dataset_name: "train"})
        self.summary_writer.add_summary(summary, self.training_step)

    def evaluate(self, sentence_lens, form_ids, forms, form_lens, lemma_ids, lemmas, lemma_lens):
        accuracy, summary = self.session.run([self.accuracy, self.summary],
                                             {self.sentence_lens: sentence_lens,
                                              self.form_ids: form_ids, self.forms: forms, self.form_lens: form_lens,
                                              self.lemma_ids: lemma_ids, self.lemmas: lemmas, self.lemma_lens: lemma_lens,
                                              self.dataset_name: "dev"})
        self.summary_writer.add_summary(summary, self.training_step)
        return accuracy

    def predict(self, sentence_lens, form_ids, forms, form_lens):
        return self.session.run(self.predictions,
                                {self.sentence_lens: sentence_lens,
                                 self.form_ids: form_ids, self.forms: forms, self.form_lens: form_lens})


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
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=100, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    print("Loading the data.", file=sys.stderr)
    data_train = morpho_dataset.MorphoDataset(args.data_train, add_bow_eow=True)
    data_dev = morpho_dataset.MorphoDataset(args.data_dev, add_bow_eow=True, train=data_train)
    data_test = morpho_dataset.MorphoDataset(args.data_test, add_bow_eow=True, train=data_train)
    bow_char = data_train.alphabet.index("<bow>")
    eow_char = data_train.alphabet.index("<eow>")

    # Construct the network
    print("Constructing the network.", file=sys.stderr)
    expname = "tagger-{}{}-bs{}-epochs{}".format(args.rnn_cell, args.rnn_cell_dim, args.batch_size, args.epochs)
    network = Network(rnn_cell=args.rnn_cell, rnn_cell_dim=args.rnn_cell_dim,
                      num_chars=len(data_train.alphabet), bow_char=bow_char, eow_char=eow_char,
                      logdir=args.logdir, expname=expname, threads=args.threads)

    # Train
    best_dev_accuracy = 0
    test_predictions = None

    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch + 1), file=sys.stderr)
        while not data_train.epoch_finished():
            sentence_lens, form_ids, charseq_ids, charseqs, charseq_lens = \
                data_train.next_batch(args.batch_size, including_charseqs=True)
            network.train(sentence_lens, charseq_ids[data_train.FORMS], charseqs[data_train.FORMS], charseq_lens[data_train.FORMS],
                          charseq_ids[data_train.LEMMAS], charseqs[data_train.LEMMAS], charseq_lens[data_train.LEMMAS])

        sentence_lens, form_ids, charseq_ids, charseqs, charseq_lens = data_dev.whole_data_as_batch(including_charseqs=True)
        dev_accuracy = network.evaluate(sentence_lens,
                                        charseq_ids[data_train.FORMS], charseqs[data_train.FORMS], charseq_lens[data_train.FORMS],
                                        charseq_ids[data_train.LEMMAS], charseqs[data_train.LEMMAS], charseq_lens[data_train.LEMMAS])
        print("Development accuracy after epoch {} is {:.2f}.".format(epoch + 1, 100. * dev_accuracy), file=sys.stderr)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            sentence_lens, form_ids, charseq_ids, charseqs, charseq_lens = data_test.whole_data_as_batch(including_charseqs=True)
            test_predictions = network.predict(sentence_lens,
                                               charseq_ids[data_train.FORMS], charseqs[data_train.FORMS], charseq_lens[data_train.FORMS])

    # Print test predictions
    test_forms = data_test.factors[data_test.FORMS]['strings'] # We use strings instead of words, because words can be <unk>
    for i in range(len(data_test.sentence_lens)):
        for j in range(data_test.sentence_lens[i]):
            lemma = ''
            for k in range(len(test_predictions[i][j])):
                if test_predictions[i][j][k] == eow_char:
                    break
                lemma += data_test.alphabet[test_predictions[i][j][k]]
            print("{}\t{}\t_".format(test_forms[i][j], lemma))
        print()
