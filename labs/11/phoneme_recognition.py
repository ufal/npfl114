#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import timit_mfcc26_dataset

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_phones, mfcc_dim):
        with self.session.graph.as_default():
            # Inputs
            self.mfcc_lens = tf.placeholder(tf.int32, [None])
            self.mfccs = tf.placeholder(tf.float32, [None, None, mfcc_dim])
            self.phone_lens = tf.placeholder(tf.int32, [None])
            self.phones = tf.placeholder(tf.int32, [None, None])

            # TODO: Computation and training. The rest of the template assumes
            # the following variables:
            # - `losses`: vector of losses, with an element for each example in the batch
            # - `edit_distances`: vector of edit distances, with an element for each batch example

            # Training
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(tf.reduce_mean(losses), global_step=global_step, name="training")

            # Summaries
            self.current_edit_distance, self.update_edit_distance = tf.metrics.mean(edit_distances)
            self.current_loss, self.update_loss = tf.metrics.mean(losses)
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/edit_distance", self.update_edit_distance)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/edit_distance", self.current_edit_distance)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size):
        while not train.epoch_finished():
            mfcc_lens, mfccs, phone_lens, phones = train.next_batch(batch_size)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.mfcc_lens: mfcc_lens, self.mfccs: mfccs,
                              self.phone_lens: phone_lens, self.phones: phones})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            mfcc_lens, mfccs, phone_lens, phones = dataset.next_batch(batch_size)
            self.session.run([self.update_edit_distance, self.update_loss],
                             {self.mfcc_lens: mfcc_lens, self.mfccs: mfccs,
                              self.phone_lens: phone_lens, self.phones: phones})
        return self.session.run([self.current_edit_distance, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        # TODO: Predict phoneme sequences for the given dataset.


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    timit = timit_mfcc26_dataset.TIMIT("timit-mfcc26.pickle")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(timit.phones), timit.mfcc_dim)

    # Train
    for i in range(args.epochs):
        network.train_epoch(timit.train, args.batch_size)

        network.evaluate("dev", timit.dev, args.batch_size)

    # Predict test data
    with open("{}/speech_recognition_test.txt".format(args.logdir), "w") as test_file:

        # TODO: Predict phonemes for test set using network.predict(timit.test, args.batch_size)
        # and save them to `test_file`. Save the phonemes for each utterance on a single line,
        # separating them by a single space. The phonemes should be printed as strings (use
        # timit.phones to convert phoneme IDs to strings).
