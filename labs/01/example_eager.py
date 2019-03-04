#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf
tf.enable_v2_behavior()

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()

# Load data
mnist = MNIST()

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer((MNIST.H, MNIST.W, MNIST.C)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu),
    tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
])

optimizer = tf.train.AdamOptimizer() # tf.keras.optimizers.Adam in v2
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
for epoch in range(args.epochs):
    accuracy.reset_states()
    for batch in mnist.train.batches(args.batch_size):
        with tf.GradientTape() as tape:
            probabilities = model(batch["images"], training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(batch["labels"], probabilities)
            accuracy(np.expand_dims(batch["labels"], -1), probabilities) # np.expand_dims not needed in v2
        gradients = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(gradients, model.variables))
    train = accuracy.result()

    accuracy.reset_states()
    for batch in mnist.dev.batches(args.batch_size):
        probabilities = model(batch["images"], training=False)
        accuracy(np.expand_dims(batch["labels"], -1), probabilities) # np.expand_dims not needed in V2
    dev = accuracy.result()
    print("Epoch {} finished, train: {}, dev: {}".format(epoch + 1, train, dev))

accuracy.reset_states()
for batch in mnist.test.batches(args.batch_size):
    probabilities = model(batch["images"], training=False)
    accuracy(np.expand_dims(batch["labels"], -1), probabilities) # np.expand_dims not needed in V2
test = accuracy.result()
print("Test: {}".format(test))
