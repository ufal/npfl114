from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf

LABELS = 10
WIDTH = 28
HEIGHT = 28
HIDDEN = 100

class Network:
    def __init__(self, logdir, experiment, threads):
        # Construct the graph
        with tf.name_scope("inputs"):
            self.images = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            flattened_images = tf.reshape(self.images, [-1, WIDTH*HEIGHT], name="flattened_images")

        with tf.name_scope("hidden_layer"):
            hidden_layer_pre = tf.matmul(flattened_images, tf.Variable(tf.random_normal([WIDTH*HEIGHT, HIDDEN]), name="W"))
            hidden_layer_pre += tf.Variable(tf.random_normal([HIDDEN]), name="b")
            hidden_layer = tf.nn.tanh(hidden_layer_pre, name="activation")
        with tf.name_scope("output_layer"):
            output_layer = tf.matmul(hidden_layer, tf.Variable(tf.random_normal([HIDDEN, LABELS]), name="W"))
            output_layer += tf.Variable(tf.random_normal([LABELS]), name="b")

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output_layer, self.labels), name="loss")
            tf.scalar_summary("training/loss", loss)
        with tf.name_scope("train"):
            self.training = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope("accuracy"):
            predictions = tf.argmax(output_layer, 1, name="predictions")
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, self.labels), tf.float32), name="accuracy")
            tf.scalar_summary("training/accuracy", accuracy)

        # Summaries
        self.summaries = {'training': tf.merge_all_summaries() }
        for dataset in ["dev", "test"]:
            self.summaries[dataset] = tf.scalar_summary(dataset + "/accuracy", accuracy)

        # Create the session
        self.session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                        intra_op_parallelism_threads=args.threads))

        self.session.run(tf.initialize_all_variables())
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.train.SummaryWriter("{}/{}-{}".format(logdir, timestamp, experiment), graph=self.session.graph, flush_secs=10)
        self.steps = 0

    def train(self, images, labels):
        self.steps += 1
        _, summary = self.session.run([self.training, self.summaries['training']], {self.images: images, self.labels: labels})
        self.summary_writer.add_summary(summary, self.steps)

    def evaluate(self, dataset, images, labels):
        summary = self.summaries[dataset].eval({self.images: images, self.labels: labels}, self.session)
        self.summary_writer.add_summary(summary, self.steps)


if __name__ == '__main__':
    # Fix random seed
    np.random.seed(42)
    tf.set_random_seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=50, type=int, help='Batch size.')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs.')
    parser.add_argument('--logdir', default="logs", type=str, help='Logdir name.')
    parser.add_argument('--exp', default="2-mnist-annotated-graph", type=str, help='Experiment name.')
    parser.add_argument('--threads', default=1, type=int, help='Maximum number of threads to use.')
    args = parser.parse_args()

    # Load the data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("mnist_data/", reshape=False)

    # Construct the network
    network = Network(logdir=args.logdir, experiment=args.exp, threads=args.threads)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
        network.evaluate("test", mnist.test.images, mnist.test.labels)
