#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf

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
    tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
    tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu),
    tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
])

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
for epoch in range(args.epochs):
    accuracy.reset_states()
    for batch in mnist.train.batches(args.batch_size):
        with tf.GradientTape() as tape:
            probabilities = model(batch["images"], training=True)
            loss = loss_fn(batch["labels"], probabilities)
            accuracy(batch["labels"], probabilities)
        gradients = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(gradients, model.variables))
    train = accuracy.result()

    accuracy.reset_states()
    for batch in mnist.dev.batches(args.batch_size):
        probabilities = model(batch["images"], training=False)
        accuracy(batch["labels"], probabilities)
    dev = accuracy.result()
    print("Epoch {} finished, train: {}, dev: {}".format(epoch + 1, train, dev))

accuracy.reset_states()
for batch in mnist.test.batches(args.batch_size):
    probabilities = model(batch["images"], training=False)
    accuracy(batch["labels"], probabilities)
test = accuracy.result()
print("Test: {}".format(test))
