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

            # Encoder
            def encoder(image):
                # TODO: Define an encoder as a sequence of:
                # - flattening layer
                # - dense layer with 500 neurons and ReLU activation
                # - dense layer with 500 neurons and ReLU activation
                #
                # Using the last hidden layer output, produce two vectors of size self.z_dim,
                # using two dense layers without activation, the first being `z_mean` and the second
                # `z_log_variance`.
                return z_mean, z_log_variance

            z_mean, z_log_variance = encoder(self.images)

            # TODO: Compute `z_sd` as a standard deviation from `z_log_variance`, by passing
            # `z_log_variance` / 2 to an exponential function.

            # TODO: Compute `epsilon` as a random normal noise, with a shape of `z_mean`.

            # TODO: Compute `self.z` by drawing from normal distribution with
            # mean `z_mean` and standard deviation `z_sd` (utilizing the `epsilon` noise).

            # Decoder
            def decoder(z):
                # TODO: Define a decoder as a sequence of:
                # - dense layer with 500 neurons and ReLU activation
                # - dense layer with 500 neurons and ReLU activation
                # - dense layer with as many neurons as there are pixels in an image
                #
                # Consider the output of the last hidden layer to be the logits of
                # individual pixels. Reshape them into a correct shape for a grayscale
                # image of size self.WIDTH x self.HEIGHT and return them.

            generated_logits = decoder(self.z)

            # TODO: Define `self.generated_images` as generated_logits passed through a sigmoid.

            # Loss and training

            # TODO: Define `reconstruction_loss` as a sigmoid cross entropy
            # loss of `self.images` and `generated_logits`.

            # TODO: Define `latent_loss` as a mean of KL-divergences of normal distributions
            # N(z_mean, z_sd) and N(0, 1), utilizing `tf.distributions.kl_divergence`
            # and `tf.distributions.Normal`.

            # TODO: Define `self.loss` as a weighted sum of
            # reconstruction_loss (weight is the number of pixels in an image)
            # and latent_loss (weight is the dimensionality of the latent variable z).

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(self.loss, global_step=global_step, name="training")

            # Summaries
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries = [tf.contrib.summary.scalar("vae/loss", self.loss),
                                  tf.contrib.summary.scalar("vae/reconstruction_loss", reconstruction_loss),
                                  tf.contrib.summary.scalar("vae/latent_loss", latent_loss)]

            self.generated_image_data = tf.placeholder(tf.float32, [None, None, 1])
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.generated_image_summary = tf.contrib.summary.image("vae/generated_image",
                                                                        tf.expand_dims(self.generated_image_data, axis=0))

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images):
        return self.session.run([self.training, self.summaries, self.loss], {self.images: images})[-1]

    def generate(self):
        GRID = 20

        def sample_z(batch_size):
            return np.random.normal(size=[batch_size, self.z_dim])

        # Generate GRIDxGRID images
        random_images = self.session.run(self.generated_images, {self.z: sample_z(GRID * GRID)})

        # Generate GRIDxGRID interpolated images
        if self.z_dim == 2:
            # Use 2D grid for sampled Z
            starts = np.stack([-2 * np.ones(GRID), np.linspace(-2, 2, GRID)], -1)
            ends = np.stack([2 * np.ones(GRID), np.linspace(-2, 2, GRID)], -1)
        else:
            # Generate random Z
            starts, ends = sample_z(GRID), sample_z(GRID)
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
