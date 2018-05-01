#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    HEIGHT, WIDTH = 28, 28

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        self.z_dim = args.z_dim

        with self.session.graph.as_default():
            if args.recodex:
                tf.get_variable_scope().set_initializer(tf.glorot_uniform_initializer(seed=42))

            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1])
            self.z = tf.placeholder(tf.float32, [None, self.z_dim])

            # Generator
            def generator(z):
                # TODO: Define a generator as a sequence of:
                # - batch normalized dense layer with 1024 neurons and ReLU activation
                # - batch normalized dense layer with 7 * 7 * 64 neurons and ReLU activation
                # - change shape to a batch of images with size 7 x 7 and 64 channels
                # - batch normalized conv2d_transpose with 32 output channels, kernel size 5,
                #   stride 2, "same" padding and ReLU activation
                # - (non-normalized) conv2d_transpose with 1 output channel, kernel size 5,
                #   stride 2, "same" padding and sigmoid activation
                # Return the result.
                #
                # Note that batch normalization should be used on inputs
                # without bias (`use_bias=False`) and that activation should be
                # applied after the batch normalization. Also use `training=True`
                # all the time (i.e., never use the saved estimates of moments)
                # in the batch normalization.

            with tf.variable_scope("generator"):
                # TODO(GAN): Define `self.generated_images` as a result of `generator` applied to `self.z`.

            # Discriminator
            def discriminator(image):
                # TODO: Define a discriminator as a sequence of:
                # - batch normalized conv2d with 32 output channels, kernel size 5,
                #   "same" padding and ReLU activation
                # - max pooling layer with kernel size 2 and stride 2
                # - batch normalized conv2d with 64 output channels, kernel size 5,
                #   "same" padding and ReLU activation
                # - max pooling layer with kernel size 2 and stride 2
                # - flattening layer
                # - batch normalized dense layer with 1024 neurons and ReLU activation
                # - (non-normalized) dense layer with 1 neuron without activation.
                #
                # Consider the last hidden layer output to be the logit of whether the input
                # images comes from real data. Change its shape to remove the last dimension
                # (i.e., [batch_size] instead of [batch_size, 1]) and return it.
                #
                # Same considerations as in `generator` regarding the batch normalization apply.

            with tf.variable_scope("discriminator"):
                # TODO(GAN): Define `discriminator_logit_real` as a result of
                # `discriminator` applied to `self.images`.

            with tf.variable_scope("discriminator", reuse = True):
                # TODO(GAN): Define `discriminator_logit_fake` as a result of
                # `discriminator` applied to `self.generated_images`.
                #
                # TODO(GAN): Note the discriminator is called in the same variable
                # scope as several lines above -- it will try to utilize the
                # same variables. In order to allow reusing them, we need to explicitly
                # pass the `reuse=True` flag.

            # Losses
            # TODO(GAN): Define `self.discriminator_loss` as a sum of
            # - sigmoid cross entropy loss with gold labels of ones (1.0) and discriminator_logit_real
            # - sigmoid cross entropy loss with gold labels of zeros (0.0) and discriminator_logit_fake

            # TODO(GAN): Define `self.generator_loss` as a sigmoid cross entropy
            # loss with gold labels of ones (1.0) and discriminator_logit_fake.

            # Training
            global_step = tf.train.create_global_step()
            # TODO(GAN): Create `self.discriminator_training` as an AdamOptimizer.minimize
            # for discriminator_loss and variables in "discriminator" namespace.
            # Do *not* pass global_step as argument to AdamOptimizer.minimize.

            # TODO(GAN): Create `self.generator_training` as an AdamOptimizer.minimize
            # for generator_loss and variables in "generator" namespace.
            # This time *do* pass global_step as argument to AdamOptimizer.minimize.

            # Summaries
            discriminator_accuracy = tf.reduce_mean(tf.to_float(tf.concat([
                tf.greater(discriminator_logit_real, 0), tf.less(discriminator_logit_fake, 0)], axis=0)))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.discriminator_summary = [tf.contrib.summary.scalar("gan/discriminator_loss", self.discriminator_loss),
                                              tf.contrib.summary.scalar("gan/discriminator_accuracy", discriminator_accuracy)]
                self.generator_summary = tf.contrib.summary.scalar("gan/generator_loss", self.generator_loss)

            self.generated_image_data = tf.placeholder(tf.float32, [None, None, 1])
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.generated_image_summary = tf.contrib.summary.image("gan/generated_image",
                                                                        tf.expand_dims(self.generated_image_data, axis=0))

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def sample_z(self, batch_size):
        # TODO(GAN): Return uniform random noise in -1, 1 range using `np.random.uniform`
        # call, with shape [batch_size, self.z_dim].

    def train(self, images):
        # TODO(GAN): In first self.session.run, evaluate self.discriminator_training,
        # self.discriminator_summary and self.discriminator_loss using
        # `images` as `self.images` and noise sampled with `self.sample_z` as `self.z`.

        # TODO(GAN): In second self.session.run, evaluate self.generator_training,
        # self.generator_summary and self.generator_loss using
        # noise sampled with `self.sample_z` as `self.z`.

        # TODO(GAN): Return the sum of evaluated self.discriminator_loss and self.generator_loss.

    def generate(self):
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.session.run(self.generated_images, {self.z: self.sample_z(GRID * GRID)})

        # Generate GRIDxGRID interpolated images
        if self.z_dim == 2:
            # Use 2D grid for sampled Z
            starts = np.stack([-np.ones(GRID), np.linspace(-1, 1, GRID)], -1)
            ends = np.stack([np.ones(GRID), np.linspace(-1, 1, GRID)], -1)
        else:
            # Generate random Z
            starts, ends = self.sample_z(GRID), self.sample_z(GRID)
        interpolated_z = []
        for i in range(GRID):
            interpolated_z.extend(starts[i] + (ends[i] - starts[i]) * np.expand_dims(np.linspace(0, 1, GRID), -1))
        interpolated_images = self.session.run(self.generated_images, {self.z: interpolated_z})

        # Stack the random images, then an empty row, and finally interpolated imates
        image = np.concatenate(
            [np.concatenate(list(images), axis=1) for images in np.split(random_images, GRID)] +
            [np.zeros([self.HEIGHT, self.WIDTH * GRID, 1])] +
            [np.concatenate(list(images), axis=1) for images in np.split(interpolated_images, GRID)], axis=0)
        self.session.run(self.generated_image_summary, {self.generated_image_data: image})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--dataset", default="mnist-data", type=str, help="Dataset [fashion|cifar-cars|mnist-data].")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="ReCodEx mode.")
    parser.add_argument("--recodex_validation_size", default=None, type=int, help="Validation size in ReCodEx mode.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
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
    if args.recodex:
        data = mnist.input_data.read_data_sets(".", reshape=False, validation_size=args.recodex_validation_size, seed=42)
    elif args.dataset == "fashion":
        data = mnist.input_data.read_data_sets("fashion", reshape=False, validation_size=0, seed=42,
                                               source_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/")
    elif args.dataset == "cifar-cars":
        data = mnist.input_data.read_data_sets("cifar-cars", reshape=False, validation_size=0, seed=42,
                                            source_url="https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/cifar-cars/")
    else:
        data = mnist.input_data.read_data_sets(args.dataset, reshape=False, validation_size=0, seed=42)


    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        loss = 0
        while data.train.epochs_completed == i:
            images, _ = data.train.next_batch(args.batch_size)
            loss += network.train(images)
        print("{:.2f}".format(loss))

        network.generate()
