#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import decoder
from morpho_dataset import MorphoDataset

class Network:
    def __init__(self, args, num_source_chars, num_target_chars):
        # TODO: Define a suitable model.

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def train(self, pdt, args):
        # TODO: Train the network on a given dataset.
        raise NotImplementedError()

    def predict(self, dataset, args):
        # TODO: Predict method should return a list, each element corresponding
        # to one sentence. Each sentence should be a list/np.ndarray
        # containing lemmas for every sentence word, the lemma being a list
        # of _indices_ of predicted characters, ended by a
        # MorphoDataset.Factor.EOW.

        # Please note that `predict_batch` from lemmatizer_{noattn,attn} assignments
        # returns flat list of all non-padding lemmas in the batch. Therefore, you
        # need to use the `dataset.sentence_lens` to reconstruct original sentences
        # from the flattened representation.

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
    morpho = MorphoDataset("czech_pdt")

    # Create the network and train
    network = Network(args,
                      num_source_chars=len(morpho.train.data[morpho.train.FORMS].alphabet),
                      num_target_chars=len(morpho.train.data[morpho.train.LEMMAS].alphabet))
    network.train(morpho, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "lemmatizer_competition_test.txt"
    if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, sentence in enumerate(network.predict(morpho.test, args)):
            for j in range(len(morpho.test.data[morpho.test.FORMS].word_strings[i])):
                lemma = []
                for c in map(int, sentence[j]):
                    if c == MorphoDataset.Factor.EOW: break
                    lemma.append(morpho.test.data[morpho.test.LEMMAS].alphabet[c])

                print(morpho.test.data[morpho.test.FORMS].word_strings[i][j],
                      "".join(lemma),
                      morpho.test.data[morpho.test.TAGS].word_strings[i][j],
                      sep="\t", file=out_file)
            print(file=out_file)
