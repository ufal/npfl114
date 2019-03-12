#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--activation", default="none", type=str, help="Activation function.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--layers", default=1, type=int, help="Number of layers.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()

# Fix random seeds
np.random.seed(42)
tf.random.set_random_seed(42)
if args.recodex:
    tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
tf.keras.backend.set_session(tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                              intra_op_parallelism_threads=args.threads)))

# Create logdir name
args.logdir = "logs/{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
)

# Load data
mnist = MNIST()

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer((MNIST.H, MNIST.W, MNIST.C)),
    tf.keras.layers.Flatten(),
])

for i in range(0,args.layers):
    if args.activation == "relu":
        model.add(tf.keras.layers.Dense(args.hidden_layer, tf.nn.relu))
    elif args.activation == "tanh":
        model.add(tf.keras.layers.Dense(args.hidden_layer, tf.nn.tanh))
    elif args.activation == "sigmoid":
        model.add(tf.keras.layers.Dense(args.hidden_layer, tf.nn.sigmoid))
    else:
        model.add(tf.keras.layers.Dense(args.hidden_layer))

model.add(tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax))

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=1000)
tb_callback.on_train_end = lambda *_: None
model.fit(
    mnist.train.data["images"], mnist.train.data["labels"],
    batch_size=args.batch_size, epochs=args.epochs,
    validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
    callbacks=[tb_callback],
)

test_logs = model.evaluate(
    mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size,
)
tb_callback.on_epoch_end(1, dict(("test_" + metric, value) for metric, value in zip(model.metrics_names, test_logs)))

accuracy = 1.0 - test_logs[0]
with open("mnist_layers_activations.out", "w") as out_file:
    print("{:.2f}".format(100 * accuracy), file=out_file)
