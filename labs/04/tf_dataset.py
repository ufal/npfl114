#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

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

    # Load the data
    cifar = CIFAR10(size={"dev": 1000})

    # Create the model
    inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
    hidden = tf.keras.layers.Conv2D(16, 3, 2, "same", activation=tf.nn.relu)(inputs)
    hidden = tf.keras.layers.Conv2D(16, 3, 1, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(24, 3, 2, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(24, 3, 1, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(32, 3, 2, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(32, 3, 1, "same", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(200, activation=tf.nn.relu)(hidden)
    outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(hidden)

    # Train the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    # TODO: Create `train` and `dev` datasets by using
    # `tf.data.Dataset.from_tensor_slices` on cifar.train and cifar.dev.
    # The structure of a single example is inferred from the argument
    # of `from_tensor_slices` -- in our case we want each example to
    # be a pair of `(input_image, target_label)`, so we need to pass
    # a pair `(data["images"], data["labels"])` to `from_tensor_slices`.
    train = ...
    dev = ...

    # Simple data augmentation
    generator = tf.random.Generator.from_seed(args.seed)
    def train_augment(image, label):
        if generator.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, CIFAR10.H + 6, CIFAR10.W + 6)
        image = tf.image.resize(image, [generator.uniform([], minval=CIFAR10.H, maxval=CIFAR10.H + 12, dtype=tf.int32),
                                        generator.uniform([], minval=CIFAR10.W, maxval=CIFAR10.W + 12, dtype=tf.int32)])
        image = tf.image.crop_to_bounding_box(
            image, target_height=CIFAR10.H, target_width=CIFAR10.W,
            offset_height=generator.uniform([], maxval=tf.shape(image)[0] - CIFAR10.H + 1, dtype=tf.int32),
            offset_width=generator.uniform([], maxval=tf.shape(image)[1] - CIFAR10.W + 1, dtype=tf.int32),
        )
        return image, label

    # TODO: Now prepare the training pipeline.
    # - first use `.take(5000)` method to utilize only the first 5000 examples
    # - call `.shuffle(5000, seed=args.seed)` to shuffle the data using
    #   the given seed and a buffer of the size of the whole data
    # - call `.map(train_augment)` to perform the dataset augmentation
    # - finally call `.batch(args.batch_size)` to generate batches
    # - optionally, you might want to add `.prefetch(tf.data.AUTOTUNE)` as
    #   the last call -- it allows the pipeline to run in parallel with
    #   the training process, dynamically adjusting the number of threads
    #   to fully saturate the training process
    train = ...

    # TODO: Prepare the `dev` pipeline
    # - just use `.batch(args.batch_size)` to generate batches
    dev = ...

    # Train
    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    # Return dev set accuracy
    return logs.history["val_accuracy"][-1]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
