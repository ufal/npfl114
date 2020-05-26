#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
import transformers

from text_classification_dataset import TextClassificationDataset

class Network:
    def __init__(self, args, labels):
        # TODO: Define a suitable model.

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def train(self, omr, args):
        # TODO: Train the network on a given dataset.
        raise NotImplementedError()

    def predict(self, dataset, args):
        # TODO: Predict method should return a list/np.ndarray of the
        # predicted label indices (no probabilities/distributions).
        raise NotImplementedError()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--bert", default="bert-base-multilingual-uncased", type=str, help="BERT model.")
    parser.add_argument("--epochs", default="2:5e-5,1:2e-5", type=str, help="Number of epochs.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.epochs = [(int(epochs), float(lr)) for epochslr in args.epochs.split(",") for epochs, lr in [epochslr.split(":")]]

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # TODO: Create the BERT tokenizer to `tokenizer` variable

    # TODO: Load the data, using a correct `tokenizer` argument, which
    # should be a callable that given a sentence in a string produces
    # a list/np.ndarray of token integers.
    facebook = TextClassificationDataset("czech_facebook", tokenizer=...)

    # Create the network and train
    network = Network(args, len(facebook.train.LABELS))
    network.train(facebook, args)

    # Generate test set annotations, but to allow parallel execution, create it
    # in in args.logdir if it exists.
    out_path = "sentiment_analysis_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for label in network.predict(facebook.test, args):
            print(facebook.test.LABELS[label], file=out_file)
