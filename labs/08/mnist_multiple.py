#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network:
    def __init__(self, args):
        # TODO: Add a `self.model` which has two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].

        # It then passes each input image through the same network (with shared weights), performing
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        # - flattening layer
        # - fully connected layer with 200 neurons and ReLU activation
        # obtaining a 200-dimensional feature representation of each image.

        # Then, it produces three outputs:
        # - classify the computed representation of the first image using a densely connected layer
        #   into 10 classes;
        # - classify the computed representation of the second image using the
        #   same connected layer (with shared weights) into 10 classes;
        # - concatenate the two 200-dimensional image representations, process
        #   them using another fully connected layer with 200 neurons and ReLU,
        #   and finally compute one output with `tf.nn.sigmoid` activation (the
        #   goal is to predict if the first digit is larger than the second)

        # Train the outputs using SparseCategoricalCrossentropy for the first two inputs
        # and BinaryCrossentropy for the third one, utilizing Adam with default arguments.
        raise NotImplementedError()

    @staticmethod
    def _prepare_batches(batches_generator):
        batches = []
        for batch in batches_generator:
            batches.append(batch)
            if len(batches) == 2:
                # TODO: yield suitable data for our task using two original
                # batches (batches[0] and batches[1]).
                yield (model_inputs, model_targets)
                batches.clear()

    def train(self, mnist, args):
        for epoch in range(args.epochs):
            # TODO: Train for one epoch using `model.train_on_batch` for each batch.
            for batch in self._prepare_batches(mnist.train.batches(args.batch_size)):
                pass

            # Print development evaluation
            print("Dev {}: directly predicting: {:.4f}, comparing digits: {:.4f}".format(epoch + 1, *self.evaluate(mnist.dev, args)))

    def evaluate(self, dataset, args):
        # TODO: Assuming the goal of the model is to predict whether
        # the first digit is larger than the second, return two accuracies:
        # - the first is `direct_accuracy`, which is the accuracy of the
        #   model's direct prediction (i.e., its third output);
        # - the second is `undirect_accuracy`, which is computed by
        #   comparing the predicted labels (i.e., the first and second output).
        for inputs, targets in self._prepare_batches(dataset.batches(args.batch_size)):
            pass

        return direct_accuracy, indirect_accuracy


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)
    with open("mnist_multiple.out", "w") as out_file:
        direct, indirect = network.evaluate(mnist.test, args)
        print("{:.2f} {:.2f}".format(100 * direct, 100 * indirect), file=out_file)
