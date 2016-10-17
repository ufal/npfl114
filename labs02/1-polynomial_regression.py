from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

class Network:
    def __init__(self, model_degree, learning_rate, threads):
        # Construct the graph
        self.xs = tf.placeholder(tf.float32, [None])
        self.ys = tf.placeholder(tf.float32, [None])

        self.predictions = tf.zeros_like(self.xs)
        for i in range(0,model_degree+1):
            self.predictions += tf.pow(self.xs, i) * tf.Variable(tf.random_normal([]))

        loss = tf.reduce_mean(tf.pow(self.predictions - self.ys, 2))
        self.training = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Create the session
        self.session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                        intra_op_parallelism_threads=args.threads))

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
    parser.add_argument('--data_degree', default=5, type=int, help='Polynomial degree of data.')
    parser.add_argument('--data_points', default=100, type=int, help='Data points.');
    parser.add_argument('--model_degree', default=5, type=int, help='Polynomial degree of model.')
    parser.add_argument('--learning_rate', default=0.01, type=int, help='Learning rate.')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs.')
    parser.add_argument('--threads', default=1, type=int, help='Maximum number of threads to use.')
    args = parser.parse_args()

    xs = np.linspace(0, 1, args.data_points)

    ys = np.ones_like(xs)
    for i in range(args.data_degree):
        ys *= np.e * (xs - i / (args.data_degree - 1))
    ys += np.random.uniform(-0.1, 0.1, args.data_points)

    plt.ion()
    plt.scatter(xs, ys)
    plt.draw()

    # Construct the network
    network = Network(model_degree=args.model_degree, learning_rate=args.learning_rate, threads=args.threads)

    for i in range(args.epochs):
        permutation = np.random.permutation(args.data_points)
        while permutation.size:
            batch_indices = permutation[:args.batch_size]
            permutation = permutation[args.batch_size:]
            network.train(xs[batch_indices], ys[batch_indices])

        if i % (args.epochs/10) == 0 or i+1 == args.epochs:
            predictions = network.predict(xs)
            plt.plot(xs, predictions, 'k', alpha=i/(args.epochs-1))
            plt.draw()

    plt.waitforbuttonpress()
