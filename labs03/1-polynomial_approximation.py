from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf

import math

import matplotlib.pyplot as plt

class Network:
    def __init__(self, layers, layer_size, threads):
        # Construct the graph
        self.xs = tf.placeholder(tf.float32, [None])
        self.ys = tf.placeholder(tf.float32, [None])

        hidden = tf.contrib.layers.fully_connected(tf.reshape(self.xs, [-1,1]), num_outputs=layer_size, activation_fn=tf.nn.relu)
        for i in range(1,layers):
            hidden = tf.contrib.layers.fully_connected(hidden, num_outputs=layer_size, activation_fn=tf.nn.relu)
        self.predictions = tf.contrib.layers.fully_connected(hidden, num_outputs=1, activation_fn=None)

        loss = tf.reduce_mean(tf.pow(self.predictions - self.ys, 2))
        self.training = tf.train.AdamOptimizer().minimize(loss)

        # Create the session
        self.session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                        intra_op_parallelism_threads=threads))

        self.session.run(tf.initialize_all_variables())

    def train(self, xs, ys):
        self.session.run(self.training, {self.xs: xs, self.ys: ys})

    def predict(self, xs):
        return self.predictions.eval({self.xs: xs}, self.session)

if __name__ == '__main__':
    # Fix random seed
    np.random.seed(42)
    tf.set_random_seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--degree', default=5, type=int, help='Polynomial degree of data.')
    parser.add_argument('--points', default=100, type=int, help='Data points.');
    parser.add_argument('--layers', default=2, type=int, help='Data points.');
    parser.add_argument('--layer_size', default=30, type=int, help='Data points.');
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
    parser.add_argument('--epochs', default=400, type=int, help='Number of epochs.')
    parser.add_argument('--threads', default=1, type=int, help='Maximum number of threads to use.')
    args = parser.parse_args()

    xs = np.linspace(0, 1, args.points)

    ys = np.ones_like(xs)
    for i in range(args.degree):
        ys *= 3 * (xs - i / (args.degree - 1))
    ys += np.random.uniform(-0.1, 0.1, args.points)

    plt.ion()
    plt.scatter(xs, ys)
    plt.draw()

    # Construct the network
    network = Network(layers=args.layers, layer_size=args.layer_size, threads=args.threads)

    for i in range(args.epochs):
        permutation = np.random.permutation(args.points)
        while permutation.size:
            batch_indices = permutation[:args.batch_size]
            permutation = permutation[args.batch_size:]
            network.train(xs[batch_indices], ys[batch_indices])

        if i % (args.epochs/10) == 0 or i+1 == args.epochs:
            predictions = network.predict(xs)
            plt.plot(xs, predictions, 'k', alpha=i/(args.epochs-1))
            plt.draw()

    plt.waitforbuttonpress()
