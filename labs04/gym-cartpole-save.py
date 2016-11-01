from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf

class Network:
    OBSERVATIONS = 4

    def __init__(self, threads=1, logdir=None, expname=None, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        if logdir:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.summary_writer = tf.train.SummaryWriter(("{}/{}-{}" if expname else "{}/{}").format(logdir, timestamp, expname), flush_secs=10)
        else:
            self.summary_writer = None

    def construct(self):
        with self.session.graph.as_default():
            self.observations = tf.placeholder(tf.float32, [None, self.OBSERVATIONS], name="observations")

            # TODO: The real model should be here, currently only a random guess
            batch_size = tf.shape(self.observations)[0]
            self.action = tf.random_uniform([batch_size], minval=0, maxval=2, dtype=tf.int32, name="action")

            # Global step
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

            # Construct the saver
            tf.add_to_collection("end_points/observations", self.observations)
            tf.add_to_collection("end_points/action", self.action)
            self.saver = tf.train.Saver(max_to_keep=None)

            # Initialize the variables
            self.session.run(tf.initialize_all_variables())

        # Finalize graph and log it if requested
        self.session.graph.finalize()
        if self.summary_writer:
            self.summary_writer.add_graph(self.session.graph)

    # Save the graph
    def save(self, path):
        self.saver.save(self.session, path)


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="", type=str, help="Logdir name.")
    parser.add_argument("--exp", default="1-gym-save", type=str, help="Experiment name.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Construct the network
    network = Network(threads=args.threads, logdir=args.logdir, expname=args.exp)
    network.construct()

    # TODO: Train the network

    # Save the network
    network.save("1-gym-random")
