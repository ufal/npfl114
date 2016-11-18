# The `resnet_v1_50.ckpt` can be downloaded from http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as tf_slim
import tensorflow.contrib.slim.nets

import imagenet_classes

class Network:
    WIDTH = 224
    HEIGHT = 224
    CLASSES = 1000

    def __init__(self, checkpoint, threads):
        # Create the session
        self.session = tf.Session(graph = tf.Graph(), config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                            intra_op_parallelism_threads=threads))

        with self.session.graph.as_default():
            # Construct the model
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 3])

            with tf_slim.arg_scope(tf_slim.nets.resnet_v1.resnet_arg_scope(is_training=False)):
                resnet, _ = tf_slim.nets.resnet_v1.resnet_v1_50(self.images, num_classes = self.CLASSES)

            self.predictions = tf.argmax(tf.squeeze(resnet, [1, 2]), 1)

            # Load the checkpoint
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, checkpoint)

            # JPG loading
            self.jpeg_file = tf.placeholder(tf.string, [])
            self.jpeg_data = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(self.jpeg_file), channels=3), self.HEIGHT, self.WIDTH)

    def load_jpeg(self, jpeg_file):
        return self.session.run(self.jpeg_data, {self.jpeg_file: jpeg_file})

    def predict(self, image):
        return self.session.run(self.predictions, {self.images: [image]})[0]

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=str, nargs='+', help="Image files.")
    parser.add_argument("--checkpoint", default="resnet_v1_50.ckpt", type=str, help="Name of ResNet50 checkpoint.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the network
    network = Network(args.checkpoint, args.threads)

    # Process the images
    for image_file in args.images:
        image_data = network.load_jpeg(image_file)
        prediction = network.predict(image_data)
        print("Image {}: class {}-{}".format(image_file, prediction, imagenet_classes.imagenet_classes[prediction]))
