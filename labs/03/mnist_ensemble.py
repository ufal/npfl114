#!/usr/bin/env python3
import argparse
import os
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[100], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--models", default=3, type=int, help="Number of models.")
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

    # Load data
    mnist = MNIST()

    # Create models
    models = []
    for model in range(args.models):
        if args.recodex:
            tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed + model)
            tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed + model)
            tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed + model)

        models.append(tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
        ] + [tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu) for hidden_layer in args.hidden_layers] + [
            tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
        ]))

        models[-1].compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        print("Training model {}: ".format(model + 1), end="", file=sys.stderr, flush=True)
        models[-1].fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs, verbose=0
        )
        print("Done", file=sys.stderr)

    individual_accuracies, ensemble_accuracies = [], []
    for model in range(args.models):
        # TODO: Compute the accuracy on the dev set for
        # the individual `models[model]`.
        individual_accuracy = None

        # TODO: Compute the accuracy on the dev set for
        # the ensemble `models[0:model+1].
        #
        # Generally you can choose one of the following approaches:
        # 1) Use Keras Functional API and construct a `tf.keras.Model`
        #    which averages the models in the ensemble (using for example
        #    `tf.keras.layers.Average` of manually with `tf.math.reduce_mean`).
        #    Then you can compile the model with the required metric (without
        #    an optimizer and a loss) and use `model.evaluate`.
        # 2) Manually perform the averaging (using TF or NumPy). In this case
        #    you do not need to construct Keras ensemble model at all,
        #    and instead call `model.predict` on individual models and
        #    average the results. To measure accuracy, either do it completely
        #    manually or use `tf.metrics.SparseCategoricalAccuracy`.
        ensemble_accuracy = None

        # Store the accuracies
        individual_accuracies.append(individual_accuracy)
        ensemble_accuracies.append(ensemble_accuracy)
    return individual_accuracies, ensemble_accuracies

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    individual_accuracies, ensemble_accuracies = main(args)
    for model, (individual_accuracy, ensemble_accuracy) in enumerate(zip(individual_accuracies, ensemble_accuracies)):
        print("Model {}, individual accuracy {:.2f}, ensemble accuracy {:.2f}".format(
            model + 1, 100 * individual_accuracy, 100 * ensemble_accuracy))
