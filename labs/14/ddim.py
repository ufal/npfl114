#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
tf.get_logger().addFilter(lambda m: "Analyzer.lamba_check" not in m.getMessage())  # Avoid pesky warning

from image64_dataset import Image64Dataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--channels", default=32, type=int, help="CNN channels in the first stage.")
parser.add_argument("--dataset", default="oxford_flowers102", type=str, help="Image64 dataset to use.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--ema", default=0.999, type=float, help="Exponential moving average momentum.")
parser.add_argument("--epoch_batches", default=1_000, type=int, help="Batches per epoch.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--loss", default="MeanAbsoluteError", type=str, help="Loss object to use.")
parser.add_argument("--plot_each", default=None, type=int, help="Plot generated images every such epoch.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--sampling_steps", default=50, type=int, help="Sampling steps.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--stage_blocks", default=2, type=int, help="ResNet blocks per stage.")
parser.add_argument("--stages", default=4, type=int, help="Stages to use.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


# The diffusion model architecture building blocks.
class SinusoidalEmbedding(tf.keras.layers.Layer):
    """Sinusoidal embeddings used to embed the current noise rate."""
    def __init__(self, dim, *args, **kwargs):
        assert dim % 2 == 0  # The `dim` needs to be even to have the same number of sin&cos.
        super().__init__(*args, **kwargs)
        self.dim = dim

    def call(self, inputs):
        # TODO: Compute the sinusoidal embeddings of the inputs in `[0, 1]` range.
        # The `inputs` have shape `[..., 1]`, and the embeddings should have
        # a shape `[..., self.dim]`, where for `0 <= i < self.dim/2`,
        # - the value on index `[..., i]` should be
        #     `sin(2 * pi * inputs / 20 ** (2 * i / self.dim))`
        # - the value on index `[..., self.dim/2 + i]` should be
        #     `cos(2 * pi * inputs / 20 ** (2 * i / self.dim))`
        raise NotImplementedError()


def ResidualBlock(inputs, width, noise_embeddings):
    """A residual block with two BN+Swish+3x3Conv, adding noise embeddings in the middle."""
    # TODO: Compute the residual connection. If the number of filters
    # in the input is the same as `width`, use unmodified `inputs`; otherwise,
    # pass it through a 1x1 convolution with `width` filters.
    residual = ...

    # TODO: Pass `inputs` through a BatchNormalization, Swish activation, and 3x3 convolution
    # with "same" padding and no bias (because it will be processed by a later batch normalization).
    hidden = ...

    # TODO: Pass `noise_embeddings` through a dense layer with `width` outputs and Swish
    # activation, and add it to `hidden`.
    hidden += ...

    # TODO: Pass `hidden` through another BatchNormalization, Swish activation, and 3x3 convolution
    # with "same" padding and no bias. Furthermore, initialize the kernel of the convolution to all
    # zeros, so that after initialization, the whole residual block is an identity.
    hidden = ...

    hidden += residual
    return hidden


# The DDIM model
class DDIM(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, data: tf.data.Dataset) -> None:
        super().__init__()

        # Create the network inputs.
        images = tf.keras.layers.Input([Image64Dataset.H, Image64Dataset.W, Image64Dataset.C])
        noise_rates = tf.keras.layers.Input([1, 1, 1])

        # TODO: Embed noise rates using the `SinusoidalEmbedding` with `args.channels` dimensions.
        noise_embedding = ...

        # TODO: Process `images` using an initial 3x3 convolution with `args.channels` filters
        # and "same" padding.
        hidden = ...

        # Downscaling stages
        outputs = []
        for i in range(args.stages):
            # TODO: For `args.stage_blocks` times, pass the `hidden` through a `ResidualBlock`
            # with `args.channels << i` filters and with the `noise_embedding`, and append
            # every result to the `outputs` array.
            ...

            # TODO: Downscale `hidden` with a 3x3 convolution with stride 2,
            # `args.channels << (i + 1)` filters, and "same" padding.
            hidden = ...

        # Middle block
        # TODO: For `args.stage_blocks` times, pass the `hidden` through a `ResidualBlock`
        # with `args.channels << args.stages` filters.
        ...

        # Upscaling stages
        for i in reversed(range(args.stages)):
            # TODO: Upscale `hidden` with a 4x4 transposed convolution with stride 2,
            # `args.channels << i` filters, and "same" padding.
            hidden = ...

            # TODO: For `args.stage_blocks` times, concatenate `hidden` and `outputs.pop()`,
            # and pass the result through a `ResidualBlock` with `args.channels << i` filters.
            ...

        # Verify that all outputs have been used.
        assert len(outputs) == 0

        # TODO: Compute the final output by passing `hidden` through a
        # BatchNormalization, Swish activation, and a 3x3 convolution with
        # `Image64Dataset.C` channels and "same" padding, with kernel of
        # the convolution initialized to all zeros.
        outputs = ...

        self._network = tf.keras.Model(inputs=[images, noise_rates], outputs=outputs)

        # Create the EMA network, which will be updated by exponential moving averaging.
        self._ema_network = tf.keras.models.clone_model(self._network)

        # Create the image normalization layer and estimate the mean and variance using `data`.
        self._image_normalization = tf.keras.layers.Normalization()
        self._image_normalization.adapt(data)

        # Store required arguments for later usage.
        self._ema_momentum = args.ema
        self._seed = args.seed

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def _image_denormalization(self, images):
        """Invert the `self._image_normalization`, returning an image represented using bytes."""
        images = self._image_normalization.mean + images * self._image_normalization.variance**0.5
        images = tf.clip_by_value(images, 0, 255)
        images = tf.cast(images, tf.uint8)
        return images

    def _diffusion_rates(self, times):
        """Compute signal and noise rates for the given times."""
        starting_angle, final_angle = 0.025, np.pi / 2 - 0.025
        # TODO: For a vector of `times` in [0, 1] range, return a pair of corresponding
        # `(signal_rates, noise_rates)`. The signal and noise rates are computed as
        # cosine and sine of an angle which is a linear interpolation from `starting_angle`
        # of 0.025 rad (for time 0) to `final_angle` of pi/2 - 0.025 rad (for time 1).
        # Because we use the rates as multipliers of image batches, reshape the rates
        # to a shape `[batch_size, 1, 1, 1]`, assuming `times` has a shape `[batch_size]`.
        signal_rates, noise_rates = ...

        return signal_rates, noise_rates

    def train_step(self, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Perform a training step."""
        # Normalize the images so have on average zero mean and unit variance.
        images = self._image_normalization(images)
        # Generate a random noise of the same shape as the `images`.
        noises = tf.random.normal(tf.shape(images), seed=self._seed)
        # Generate a batch of times when to perform the loss computation in.
        times = tf.random.uniform(tf.shape(images)[:1], seed=self._seed)

        # TODO: Compute the signal and noise rates using the sampled `times`.
        signal_rates, noise_rates = ...

        # TODO: Compute the noisy images utilizing the computed signal and noise rates.
        noisy_images = ...

        with tf.GradientTape() as tape:
            # TODO: Predict the noise by running the `self._network` on the noisy images
            # and the noise rates. Do not forget to also pass the `training=True` argument
            # (to run batch normalizations in training regime).
            predicted_noises = ...

            # TODO: Compute loss using the `self.compiled_loss`.
            loss = ...

        # Perform an update step.
        self.optimizer.minimize(loss, self._network.trainable_variables, tape=tape)

        # Update the `self._ema_network` using exponential moving average.
        for ema_variable, variable in zip(self._ema_network.variables, self._network.variables):
            ema_variable.assign(self._ema_momentum * ema_variable + (1 - self._ema_momentum) * variable)

        return {metric.name: metric.result() for metric in self.metrics}

    def generate(self, initial_noise, steps):
        """Sample a batch of images given the `initial_noise` using `steps` steps."""
        images = initial_noise
        diffusion_process = []

        # We emply a uniformly distributed sequence of times from 1 to 0. We in fact
        # create an identical batch of them, and we also make the time of the next step
        # available in the body of the cycle, because it is needed by the DDIM algorithm.
        steps = tf.linspace(tf.ones(tf.shape(initial_noise)[0]), tf.zeros(tf.shape(initial_noise)[0]), steps + 1)

        for times, next_times in zip(steps[:-1], steps[1:]):
            # Store the current images converted to `tf.uint8` to allow denoising visualization.
            diffusion_process.append(self._image_denormalization(images))

            # TODO: Compute the signal and noise rates of the current time step.
            signal_rates, noise_rates = ...

            # TODO: Predict the noise by calling the `self._ema_network` with `training=False`.
            predicted_noises = ...

            # TODO: Compute the signal and noise rates of the next time step.
            next_signal_rates, next_noise_rates = ...

            # TODO: Predict the denoised version of `images` (i.e., the $x_0$ estimate
            # in the DDIM sampling algorithm).
            denoised_images = ...

            # TODO: Update the `images` according to the DDIM sampling algorithm.
            images = ...

        # TODO: Compute the output by passing the latest `denoised_images` through
        # the `self._image_denormalization` to obtain a `tf.uint8` representation.
        images = ...

        return images, diffusion_process


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the image data.
    images64 = Image64Dataset(args.dataset).train.map(lambda example: example["image"])

    # Create the model; the image data are used to initialize the image normalization layer.
    ddim = DDIM(args, images64)

    # Create the data pipeline.
    train = images64.repeat()
    train = train.shuffle(10 * args.batch_size, seed=args.seed)
    train = train.batch(args.batch_size)
    train = train.prefetch(tf.data.AUTOTUNE)

    # Class for sampling images and storing them to TensorBoard.
    class TBSampler:
        def __init__(self, columns: int, rows: int) -> None:
            self._columns = columns
            self._rows = rows
            self._noise = tf.random.normal(
                [columns * rows, Image64Dataset.H, Image64Dataset.W, Image64Dataset.C], seed=args.seed)

        def __call__(self, epoch, logs=None) -> None:
            # After the last epoch and every `args.plot_each` epoch, generate a sample to TensorBoard logs.
            if epoch + 1 == args.epochs or (epoch + 1) % (args.plot_each or args.epochs) == 0:
                # Generate a grid of `self._columns *  self._rows` independent samples.
                samples, _ = ddim.generate(self._noise, args.sampling_steps)
                images = tf.concat([tf.concat(list(row), axis=1) for row in tf.split(samples, self._rows)], axis=0)
                with ddim.tb_callback._train_writer.as_default(step=epoch):
                    tf.summary.image("images", images[tf.newaxis])
                # Generate gradual denoising process for `rows` samples, showing `self._columns` steps.
                steps = args.sampling_steps // self._columns + 1
                samples, process = ddim.generate(self._noise[::self._columns], steps * (self._columns - 2) + 1)
                process = tf.concat([tf.reshape(s, [-1, *s.shape[2:]]) for s in process[::steps] + [samples]], axis=1)
                with ddim.tb_callback._train_writer.as_default(step=epoch):
                    tf.summary.image("process", process[tf.newaxis])
            # After the last epoch, store statistics of the generated sample for ReCodEx to evaluate.
            if epoch + 1 == args.epochs:
                logs["sample_mean"] = tf.math.reduce_mean(tf.cast(images, tf.float32))
                logs["sample_std"] = tf.math.reduce_std(tf.cast(images, tf.float32))

    # Train the model
    ddim.compile(
        optimizer=tf.optimizers.experimental.AdamW(jit_compile=False),
        loss=getattr(tf.losses, args.loss)(),
    )
    logs = ddim.fit(train, epochs=args.epochs, steps_per_epoch=args.epoch_batches, callbacks=[
        tf.keras.callbacks.LambdaCallback(on_epoch_end=TBSampler(16, 10)), ddim.tb_callback])

    # Return the loss and sample statistics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items()}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
