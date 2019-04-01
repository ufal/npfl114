#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from fashion_masks_data import FashionMasks

# TODO: Define a suitable model in the Network class.
# A suitable starting model contains some number of shared
# convolutional layers, followed by two heads, one predicting
# the label and the other one the masks.
class Network:
    def __init__(self, args):
        pass

    def train(self, fashion_masks, args):
        pass

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

    # Fix random seeds
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

    # Load data
    fashion_masks = FashionMasks()

    # Create the network and train
    network = Network(args)
    network.train(fashion_masks, args)

    # Predict test data in args.logdir
    with open(os.path.join(args.logdir, "fashion_masks_test.txt"), "w", encoding="utf-8") as out_file:
        # TODO: Predict labels and masks on fashion_masks.test.data["images"],
        # into test_labels and test_masks (test_masks is assumed to be
        # a Numpy array with values 0/1).
        for label, mask in zip(test_labels, test_masks):
            print(label, *mask.astype(np.uint8).flatten(), file=out_file)
