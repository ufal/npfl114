#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from omr_dataset import OMRDataset

class Network:
    def __init__(self, omr, args):
        # TODO: Define a suitable model.

    def train(self, omr, args):
        # TODO: Train the network on a given dataset.
        raise NotImplementedError()

    def predict(self, dataset, args):
        # TODO: Predict method should return a list, each element corresponding
        # to one test image. Each element should be a list/np.ndarray
        # containing indices recognized marks (without padding).
        raise NotImplementedError()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

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

    # Load the data
    omr = OMRDataset()

    # Create the network and train
    network = Network(omr, args)
    network.train(omr, args)

    # Generate test set annotations, but to allow parallel execution, create it
    # in in args.logdir if it exists.
    out_path = "omr_competition_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for sequence in network.predict(omr.test, args):
            print(" ".join(omr.MARKS[mark] for mark in sequence), file=out_file)
