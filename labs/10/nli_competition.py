#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from nli_dataset import NLIDataset

class Network:
    def __init__(self, nli, args):
        # TODO: Define a suitable model.

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def train(self, nli, args):
        # TODO: Train the network on a given dataset.
        raise NotImplementedError()

    def predict(self, dataset, args):
        # TODO: Predict method should return a list/np.ndaddar, each element
        # being the predicted language for a sencence.
        raise NotImplementedError()


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    nli = NLIDataset()

    # Create the network and train
    network = Network(nli, args)
    network.train(nli, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "nli_competition_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        languages = network.predict(nli.test, args)
        for language in languages:
            print(nli.test.vocabulary("languages")[language], file=out_file)
