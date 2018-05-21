#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
# TODO: Import `eager` from `contrib` module, into `tfe`

# TODO: Enable eager mode using `enable_eager_execution`.

class Network(tfe.Network):
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, args):
        super(Network, self).__init__(name='')

        # TODO: Add layers to the Network, using `self.track_layer`.
        # Create:
        # - a Conv2D layer with 8 filters, kernel size 3 and ReLU activation
        # - a Conv2D layer with 16 filters, kernel size 3 and ReLU activation
        # - a MaxPooling2D layer with pooling size 2 and stride 2
        # - a Dense layer with 256 neurons and ReLU activation
        # - a Dense layer with self.LABELS neurons and no activation
        # - a Dropout layer with 0.5 dropout rate

        self.global_step = tf.train.create_global_step()
        self.optimizer = tf.train.AdamOptimizer()
        self.summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def call(self, inputs, training):
        # TODO: Process inputs, using the above created layers:
        # - convolution with 8 filters
        # - max pooling
        # - convolution with 16 filters
        # - max pooling
        # - flattening layer
        # - dense layer with 256 neurons
        # - dropout layer
        # - dense layer with self.LABELS neurons
        # Return the computed logits.

    def predict(self, logits):
        return tf.argmax(logits, axis=1)

    def train_epoch(self, dataset):
        # TODO: Iterate over images and labels using `tfe.Iterator` in the `dataset`.
        for ...:
            # Use `tfe.GradientTape` to store loss computation
            with ... :
                # TODO: Compute `logits` using `self(images, training=True)`
                # TODO: Compute `loss` using `tf.losses.sparse_softmax_cross_entropy`

            with self.summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                tf.contrib.summary.scalar("train/loss", loss)
                accuracy = tfe.metrics.Accuracy("train/accuracy"); accuracy(labels, self.predict(logits)); accuracy.result()

            # TODO: Compute `gradients` of `loss` with respect to `self.variables` using the `GradientTape`
            self.optimizer.apply_gradients(zip(gradients, self.variables), global_step=self.global_step)

    def evaluate(self, dataset_name, dataset):
        # Create `accuracy` metric using `tfe.metrics.Accuracy` with `dataset_name + "/accuracy"` name

        # TODO: Iterate over images and labels using `tfe.Iterator` in the `dataset`.
        for ...:
            # TODO: Compute `logits` using `self(images, training=True)`

            # TODO: Update accuracy metric using the `labels` and predictions from `logits`

        with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            # This both adds a summary and returns the result
            return accuracy.result()


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)
    tf.set_random_seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
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
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets("mnist-data", reshape=False, seed=42)

    # TODO: Create `train` dataset using `from_tensor_slices`, shuffle it using a buffer of 60000 and generate batches of size
    # args.batch_size. Note that `mnist.train.labels` must be converted to `np.int` (for example using
    # mnist.train.labels.astype(np.int) call).

    # TODO: Create `dev` and `test` datasets similarly, but without shuffling.

    # Construct the network
    network = Network(args)

    # Train
    for i in range(args.epochs):
        network.train_epoch(train)

        print("Dev acc after epoch {}: {}".format(i + 1, network.evaluate("dev", dev)))
    print("Test acc: {}".format(network.evaluate("test", test)))
