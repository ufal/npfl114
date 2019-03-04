#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2 as tf_summary

from mnist import MNIST

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, MNIST.H, MNIST.W, MNIST.C], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")

            # Computation
            hidden = tf.keras.layers.Flatten()(self.images)
            # TODO: Add `args.layers` number of hidden layers with size `args.hidden_layer`,
            # using activation from `args.activation`, allowing "none", "relu", "tanh", "sigmoid".
            # Store the results back to `hidden` variable.
            output_layer = tf.keras.layers.Dense(MNIST.LABELS)(hidden)
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            loss = tf.keras.losses.sparse_categorical_crossentropy(self.labels, output_layer, from_logits=True)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            accuracy = tf.math.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            confusion_matrix = tf.reshape(tf.confusion_matrix(self.labels, self.predictions,
                                                              weights=tf.not_equal(self.labels, self.predictions), dtype=tf.float32),
                                          [1, MNIST.LABELS, MNIST.LABELS, 1])

            summary_writer = tf_summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf_summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf_summary.scalar("train/loss", loss),
                                           tf_summary.scalar("train/accuracy", accuracy)]
            with summary_writer.as_default(), tf_summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf_summary.scalar(dataset + "/accuracy", accuracy),
                                               tf_summary.image(dataset + "/confusion_matrix", confusion_matrix)]
                    with tf.control_dependencies(self.summaries[dataset]):
                        self.summaries[dataset].append(summary_writer.flush())

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf_summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels})

    def evaluate(self, dataset, images, labels):
        self.session.run(self.summaries[dataset], {self.images: images, self.labels: labels})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", default="none", type=str, help="Activation function.")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
    parser.add_argument("--layers", default=1, type=int, help="Number of layers.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
    tf.keras.backend.set_session(tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                                  intra_op_parallelism_threads=args.threads)))

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )

    # Load the data
    mnist = MNIST()

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        for batch in mnist.train.batches(args.batch_size):
            network.train(batch["images"], batch["labels"])

        network.evaluate("dev", mnist.dev.data["images"], mnist.dev.data["labels"])
    network.evaluate("test", mnist.test.data["images"], mnist.test.data["labels"])

    # TODO: Write test accuracy as percentages rounded to two decimal places.
    with open("mnist_graph.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
