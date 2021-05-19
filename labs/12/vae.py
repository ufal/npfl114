#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--decoder_layers", default=[500,500], type=int, nargs="+", help="Decoder layers.")
parser.add_argument("--encoder_layers", default=[500,500], type=int, nargs="+", help="Encoder layers.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
# If you add more arguments, ReCodEx will keep them with your default values.

# The VAE model
class VAE(tf.keras.Model):
    def __init__(self, args):
        super().__init__()

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = tfp.distributions.Normal(tf.zeros(args.z_dim), tf.ones(args.z_dim))

        # TODO: Define `self.encoder` as a Model, which
        # - takes input images with shape [MNIST.H, MNIST.W, MNIST.C]
        # - flattens them
        # - applies len(args.encoder_layers) dense layers with ReLU activation,
        #   i-th layer with args.encoder_layers[i] units
        # - generate two outputs z_mean and z_log_variance, each passing the result
        #   of the above line through its own dense layer with args.z_dim units

        # TODO: Define `self.decoder` as a Model, which
        # - takes vectors of [args.z_dim] shape on input
        # - applies len(args.decoder_layers) dense layers with ReLU activation,
        #   i-th layer with args.decoder_layers[i] units
        # - applies output dense layer with MNIST.H * MNIST.W * MNIST.C units
        #   and a suitable output activation
        # - reshapes the output (tf.keras.layers.Reshape) to [MNIST.H, MNIST.W, MNIST.C]

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq="epoch", profile_batch=0)

    def train_step(self, images):
        with tf.GradientTape() as tape:
            # TODO: Compute z_mean and z_log_variance of given images using `self.encoder`.
            # Note that you should pass `training=True` to the `self.encoder`.

            # TODO: Sample `z` from a Normal distribution with mean `z_mean` and
            # standard deviation `exp(z_log_variance / 2)`. Start by creating
            # corresponding distribution `tfp.distributions.Normal(...)` and then
            # run the `sample(seed=self._seed)` method.
            #
            # Note that the distributions in `tfp` are already reparametrized if possible,
            # so you do not need to implement the reparametrization trick manually.
            # For a given distribution, you can use the `reparameterization_type` member
            # to check if it is reparametrized or not.

            # TODO: Decode images using `z` (also passing `training=True` to the `self.decoder`).

            # TODO: Define `reconstruction_loss` using `self.compiled_loss`

            # TODO: Define `latent_loss` as a mean of KL divergences of suitable distributions.
            # Note that the `tfp` distributions offer a method `kl_divergence`.

            # TODO: Define `loss` as a sum of the reconstruction_loss (multiplied by the number
            # of pixels in an image) and the latent_loss (multiplied by self._z_dim).

        # TODO: Run an optimizer step with respect to trainable variables of both the encoder and the decoder.

        return {"reconstruction_loss": reconstruction_loss, "latent_loss": latent_loss, "loss": loss}

    def generate(self, epoch, logs):
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.decoder(self._z_prior.sample(GRID * GRID, seed=self._seed), training=False)

        # Generate GRIDxGRID interpolated images
        if self._z_dim == 2:
            # Use 2D grid for sampled Z
            starts = tf.stack([-2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
            ends = tf.stack([2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
        else:
            # Generate random Z
            starts, ends = self._z_prior.sample(GRID, seed=self._seed), self._z_prior.sample(GRID, seed=self._seed)
        interpolated_z = tf.concat(
            [starts[i] + (ends[i] - starts[i]) * tf.expand_dims(tf.linspace(0., 1., GRID), -1) for i in range(GRID)], axis=0)
        interpolated_images = self.decoder(interpolated_z, training=False)

        # Stack the random images, then an empty row, and finally interpolated imates
        image = tf.concat(
            [tf.concat(list(images), axis=1) for images in tf.split(random_images, GRID)] +
            [tf.zeros([MNIST.H, MNIST.W * GRID, MNIST.C])] +
            [tf.concat(list(images), axis=1) for images in tf.split(interpolated_images, GRID)], axis=0)
        with self.tb_callback._train_writer.as_default(step=epoch):
            tf.summary.image("images", tf.expand_dims(image, 0))

def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST(args.dataset)

    # Create the network and train
    network = VAE(args)
    network.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.BinaryCrossentropy())
    logs = network.fit(
        mnist.train.dataset.map(lambda example: example["images"]).shuffle(mnist.train.size, args.seed).batch(args.batch_size),
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(on_epoch_end=network.generate),
            network.tb_callback,
        ]
    )

    # Return loss for ReCodEx to validate
    return logs.history["loss"][-1]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
