#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default="200", type=str, help="Hidden layer configuration.")
parser.add_argument("--models", default=3, type=int, help="Number of models.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Load data
mnist = MNIST()

# Create models
models = []
for model in range(args.models):
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42 + model)

    models.append(tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
    ] + [tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu) for hidden_layer in args.hidden_layers] + [
        tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
    ]))

    models[-1].compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="individual_accuracy")],
    )

    print("Training model {}: ".format(model + 1), end="", flush=True)
    models[-1].fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs, verbose=0
    )
    print("Done")

with open("mnist_ensemble.out", "w") as out_file:
    for model in range(args.models):
        # TODO: Compute the accuracy on the dev set for
        # the individual `models[model]`.
        individual_accuracy = None

        # TODO: Compute the accuracy on the dev set for
        # the ensemble `models[0:model+1].
        #
        # Generally you can choose one of the following approaches:
        # 1) Use Keras Functional API and construct a tf.keras.Model
        #    which averages the models in the ensemble (using
        #    tf.keras.layers.Average). Then you can compile the model
        #    with the required metric and use `model.evaluate`.
        #    Note that there is currently a bug which triggers if
        #    the model metric names collide, so your ensemble model must
        #    use different metric name than "individual_accuracy".
        # 2) Manually perform the averaging using NumPy. In this case
        #    you do not need to construct Keras ensemble model at all,
        #    and instead call `model.predict` on individual models and
        #    average the results. To measure accuracy, either do it completely
        #    manually or use tf.keras.metrics.SparseCategoricalAccuracy.
        ensemble_accuracy = None

        # Print the results.
        print("{:.2f} {:.2f}".format(100 * individual_accuracy, 100 * ensemble_accuracy), file=out_file)
