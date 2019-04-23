#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

class Network:
    def __init__(self, pdt, args):
        # TODO: Define a suitable model.

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def train(self, pdt, args):
        # TODO: Train the network on a given dataset.
        raise NotImplementedError()

    def predict(self, dataset, args):
        # TODO: Predict method should return a list, each element corresponding
        # to one sentence. Each sentence should be a list/np.ndarray
        # containing _indices_ of chosen tags (not the logits/probabilities).
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

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # Create the network and train
    network = Network(morpho, args)
    network.train(morpho, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "tagger_competition_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, sentence in enumerate(network.predict(morpho.test, args)):
            for j in range(len(morpho.test.data[morpho.test.FORMS].word_strings[i])):
                print(morpho.test.data[morpho.test.FORMS].word_strings[i][j],
                      morpho.test.data[morpho.test.LEMMAS].word_strings[i][j],
                      morpho.test.data[morpho.test.TAGS].words[sentence[j]],
                      sep="\t", file=out_file)
            print(file=out_file)
