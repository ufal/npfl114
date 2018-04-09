#!/usr/bin/env python3

# This source depends on the NASNet A Mobile network, which can be downloaded
# from http://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nasnet_a_mobile.zip.

import numpy as np
import tensorflow as tf

import imagenet_classes
import nets.nasnet.nasnet

class Network:
    WIDTH, HEIGHT = 224, 224
    CLASSES = 1000

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 3], name="images")

            # Computation
            images = 2 * (tf.image.convert_image_dtype(self.images, tf.float32) - 0.5)

            with tf.contrib.slim.arg_scope(nets.nasnet.nasnet.nasnet_mobile_arg_scope()):
                self.output_layer, _ = nets.nasnet.nasnet.build_nasnet_mobile(images, num_classes=self.CLASSES + 1, is_training=False)
            self.nasnet_saver = tf.train.Saver()

            self.predictions = tf.argmax(self.output_layer, axis=1) - 1

            # Image loading
            self.image_file = tf.placeholder(tf.string, [])
            self.image_data = tf.image.decode_image(tf.read_file(self.image_file), channels=3)
            self.image_data_resized = tf.image.resize_image_with_crop_or_pad(self.image_data, self.HEIGHT, self.WIDTH)

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            self.nasnet_saver.restore(self.session, args.checkpoint)

    def load_image(self, image_file):
        return self.session.run(self.image_data_resized, {self.image_file: image_file})

    def predict(self, image):
        return self.session.run(self.predictions, {self.images: [image]})[0]


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=str, nargs='+', help="Image files.")
    parser.add_argument("--checkpoint", default="nets/nasnet/model.ckpt", type=str, help="Checkpoint path.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Process the images
    for image_file in args.images:
        image_data = network.load_image(image_file)
        prediction = network.predict(image_data)
        print("Image {}: class {}-{}".format(image_file, prediction, imagenet_classes.imagenet_classes[prediction]))
