#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

def mnist_model(features, labels, mode, params):
    # TODO: Using features["images"], compute `logits` using:
    # - convolutional layer with 8 channels, kernel size 3 and ReLu activation
    # - max pooling layer with pool size 2 and stride 2
    # - convolutional layer with 16 channels, kernel size 3 and ReLu activation
    # - max pooling layer with pool size 2 and stride 2
    # - flattening layer
    # - dense layer with 256 neurons and ReLU activation
    # - dense layer with 10 neurons and no activation

    predictions = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # TODO: Return EstimatorSpec with `mode` and `predictions` parameters

    # TODO: Compute loss using `tf.losses.sparse_softmax_cross_entropy`.

    if mode == tf.estimator.ModeKeys.TRAIN:
        # TODO: Get optimizer class, using `params.get("optimizer", None)`.
        # TODO: Create optimizer, using `params.get("learning_rate", None)` parameter.
        # TODO: Define `train_op` as `optimizer.minimize`, with `tf.train.get_global_step` as `global_step`.
        # TODO: Return EstimatorSpec with `mode`, `loss`, `train_op` and `eval_metric_ops` arguments,
        # the latter being a dictionary with "accuracy" key and `tf.metrics.accuracy` value.

    if mode == tf.estimator.ModeKeys.EVAL:
        # TODO: Return EstimatorSpec with `mode`, `loss` and `eval_metric_ops` arguments,
        # the latter being a dictionary with "accuracy" key and `tf.metrics.accuracy` value.


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)
    tf.set_random_seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Construct the model
    model = tf.estimator.Estimator(
        model_fn=mnist_model,
        model_dir=args.logdir,
        config=tf.estimator.RunConfig(tf_random_seed=42,
                                      session_config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                                    intra_op_parallelism_threads=args.threads)),
        params={
            "optimizer": tf.train.AdamOptimizer,
            "learning_rate": 0.001,
        })

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets("mnist-data", reshape=False, seed=42)

    # Train
    for i in range(args.epochs):
        # TODO: Define input_fn using `tf.estimator.inputs.numpy_input_fn`.
        # As `x`, pass "images": mnist.train images, as `y` pass `mnist.train.labels.astype(np.int)`,
        # use specified batch_size, one epoch. Normally we would shuffle data with queue capacity 60000,
        # but random seed cannot be passed to this method; hence, do not shuffle data.
        # TODO: Train one epoch using the defined input_fn.


        # TODO: Define validation input_fn similarly, but without suffling.
        # TODO: Evaluate the validation data, using `model.evaluate` with `name="dev"` option
        # and print its return value.

    # TODO: Define input_fn for test set as for validation data.
    # TODO: Evaluate the test set using `model.evaluate` with `name="test"` option
    # and print its return value.
