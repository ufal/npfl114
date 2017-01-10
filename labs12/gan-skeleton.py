#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

class GANNetwork:
    WIDTH, HEIGHT = 28, 28

    def __init__(self, z_dim, logdir, expname, threads=1, seed=42):
        self.z_dim = z_dim
        self.steps = 0

        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_dir = "{}/{}-{}".format(logdir, timestamp, expname)
        self.summary_writer = tf.train.SummaryWriter(self.summary_dir, flush_secs=10)

        def generator(z):
            # TODO: Implement a generator, which
            # - starts with z
            # - add fully connected hidden layer of size 128 + ReLU
            # - add fully connected output layer of size HEIGHT*WIDTH, apply sigmoid
            # - reshapes the output to [batch_size, 28, 28, 1]
            return ...

        def discriminator(image):
            # TODO: Implement a discriminator, which
            # - starts with image
            # - flattens image to [batch_size, HEIGHT*WIDTH]
            # - add fully connected hidden layer of size 128 + ReLU
            # - add output layer of size 1 with no activation function
            # - reshapes the output from [batch_size, 1] to [batch_size]
            return ...

        # Construct the graph
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1])
            self.z = tf.placeholder(tf.float32, [None, z_dim])

            # Generator
            with tf.variable_scope("generator"):
                self.generated_images = generator(self.z)

            # Discriminator
            with tf.variable_scope("discriminator"):
                discriminator_logit_real = discriminator(self.images)
            with tf.variable_scope("discriminator", reuse = True):
                discriminator_logit_fake = discriminator(self.generated_images)

            # Losses
            # TODO: discriminator_loss is composed of two parts
            # - mean of sigmoid_cross_entropy_with_logits between discriminator_logit_real and all ones
            # - mean of sigmoid_cross_entropy_with_logits between discriminator_logit_fake and all zeros
            # - these parts are added together
            discriminator_loss = ...
            # TODO: generator_loss is mean of sigmoid_cross_entropy_with_logits between discriminator_logit_fake and all ones
            generator_loss = ...

            # Training
            # TODO: discriminator_training should use Adam and minimize discriminator_loss, but ONLY THE VARIABLES OF THE DISCRIMINATOR
            #   can be optimized -- use var_list argument of minimize and tf.get_collection(tf.GraphKeys.VARIABLES, "discriminator")
            self.discriminator_training = ...
            # TODO: generator_training should use Adam and minimize generator_loss, but ONLY THE VARIABLES OF THE GENERATOR can be optimized
            self.generator_training = ...

            # Summaries
            discriminator_accuracy = tf.reduce_mean([
                tf.to_float(tf.greater(discriminator_logit_real, 0)),
                tf.to_float(tf.less(discriminator_logit_fake, 0))
            ])
            self.discriminator_summary = tf.merge_summary([tf.scalar_summary("discriminator/loss", discriminator_loss),
                                                           tf.scalar_summary("discriminator/accuracy", discriminator_accuracy)])
            self.generator_summary = tf.scalar_summary("generator/loss", generator_loss)

            self.png_image_data = tf.placeholder(tf.float32)
            self.png_image = tf.image.encode_png(tf.image.convert_image_dtype(self.png_image_data, tf.uint8))

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    def sample_z(self, batch_size):
        return np.random.uniform(-1, 1, size=[batch_size, self.z_dim])

    def predict(self, z):
        return self.session.run(self.generated_images, {self.z: z})

    def train(self, images):
        # TODO: train the discriminator
        # - sample z using self.sample_z(len(images))
        # - run self.discriminator_training and collect self.discriminator_summary using self.images=images and self.z=sampled_z
        # - store the summary
        z = self.sample_z(len(images))
        _, summary = ...
        if self.steps % 100 == 0:
            self.summary_writer.add_summary(summary, self.steps)

        # TODO: train the generator
        # - sample another z using self.sample_z(len(images))
        # - run self.generator_training and collect self.generator_summary using self.z=sampled_z
        # - store the summary
        z = self.sample_z(len(images))
        _, summary = ...
        if self.steps % 100 == 0:
            self.summary_writer.add_summary(summary, self.steps)

        self.steps += 1

    def generate_images(self, n, step):
        # Generate nxn images
        random_images = self.predict(self.sample_z(n * n))

        # Generate 2n z-es and interpolate between the neighbours
        interpolated_z = []
        if self.z_dim == 2:
            # Use 2d grid for z-es
            starts, ends = np.stack([-np.ones(n), np.linspace(-1, 1, n)], -1), np.stack([np.ones(n), np.linspace(-1, 1, n)], -1)
        else:
            # Generate 2n random z-es
            starts, ends = self.sample_z(n), self.sample_z(n)
        for i in range(n):
            for w in np.linspace(0, 1, n):
                interpolated_z.append(starts[i] + (ends[i] - starts[i]) * w)
        interpolated_images = self.predict(interpolated_z)

        # Stack at first nxn random images, then 1xn empty images, and nxn interpolated images
        image = np.concatenate(
            map(lambda images: np.concatenate(list(images), axis=1), np.split(random_images, n, 0)) +
            [np.zeros([self.HEIGHT, self.WIDTH * n, 1])] +
            map(lambda images: np.concatenate(list(images), axis=1), np.split(interpolated_images, n, 0)),
            axis = 0
        )

        with open("{}/images{}.png".format(self.summary_dir, str(step).zfill(3)), "wb") as file:
            file.write(self.session.run(self.png_image, {self.png_image_data: image}))

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
    parser.add_argument("--dataset", default="mnist", type=str, help="Directory name of the dataset.")
    parser.add_argument("--batches", default=100000, type=int, help="Number of batches to train.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
    args = parser.parse_args()

    # Load the data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(args.dataset, reshape=False, validation_size=0)

    # Construct the network
    expname = "gan-dataset_{}-zdim_{}-batch_{}-batches_{}".format(args.dataset, args.z_dim, args.batch_size, args.batches)
    network = GANNetwork(args.z_dim, logdir=args.logdir, expname=expname, threads=args.threads)

    # Train
    for i in range(args.batches):
        images, _ = mnist.train.next_batch(args.batch_size)
        network.train(images)

        if i % 1000 == 0:
            network.generate_images(20, i//1000 + 1)
