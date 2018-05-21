#!/usr/bin/env python3
import tensorflow as tf

class ConvNet:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    # Create the network according to given hyperparameters.
    def __init__(self,
                 conv1_channels, conv1_kernel_size, conv1_stride, pool1_pool_size, pool1_stride,
                 conv2_channels, conv2_kernel_size, conv2_stride, pool2_pool_size, pool2_stride,
                 hidden_layer_size, batch_size, learning_rate, threads=1, seed=42):

        self.batch_size = batch_size

        # Create session and empty graph
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1])
            self.labels = tf.placeholder(tf.int64, [None])

            x = self.images
            x = tf.layers.conv2d(x, conv1_channels, conv1_kernel_size, conv1_stride, activation=tf.nn.relu)
            x = tf.layers.max_pooling2d(x, pool1_pool_size, pool1_stride)
            x = tf.layers.conv2d(x, conv2_channels, conv2_kernel_size, conv2_stride, activation=tf.nn.relu)
            x = tf.layers.max_pooling2d(x, pool2_pool_size, pool2_stride)
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, hidden_layer_size, activation=tf.nn.relu)
            logits = tf.layers.dense(x, self.LABELS)
            self.predictions = tf.argmax(logits, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, logits)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

            # Evaluation
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    # Train given the MNIST data. Fashion MNIST can be loaded using:
    # from tensorflow.examples.tutorials import mnist
    # data = mnist.input_data.read_data_sets("fashion-mnist", reshape=False, validation_size=0, seed=42,
    #                                        source_url="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/")
    def train(self, mnist, epochs):
        start = mnist.train.epochs_completed

        for i in range(epochs):
            while mnist.train.epochs_completed == start + i:
                images, labels = mnist.train.next_batch(self.batch_size)
                self.session.run(self.training, {self.images: images, self.labels: labels})

        return self.session.run(self.accuracy, {self.images: mnist.validation.images,
                                                self.labels: mnist.validation.labels})

    # Return hyperparameters and their ranges and distributions.
    def hyperparameters(self):
        return {
            "conv1_channels": Range("int", 1, 16),      # [1, ..., 16]
            "conv1_kernel_size": Range("int", 1, 3),    # [1, 2, 3]
            "conv1_stride": Range("int", 1, 3),         # [1, 2, 3]
            "pool1_pool_size": Range("int", 1, 3),      # [1, 2, 3]
            "pool1_stride": Range("int", 1, 3),         # [1, 2, 3]
            "conv2_channels": Range("int", 1, 16),      # [1, ..., 16]
            "conv2_kernel_size": Range("int", 1, 3),    # [1, 2, 3]
            "conv2_stride": Range("int", 1, 3),         # [1, 2, 3]
            "pool2_pool_size": Range("int", 1, 3),      # [1, 2, 3]
            "pool2_stride": Range("int", 1, 3),         # [1, 2, 3]
            "hidden_layer_size": Range("int", 16, 128), # [16, ..., 128]
            "batch_size": Range("int", 1, 64),          # [1, ..., 64]
            "learning_rate": Range("log_float", 0.01, 0.001), # logarithmically distributed real number in range [0.01-0.0001]
        }
