#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub # Note: you need to install tensorflow_hub

import imagenet_classes

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+", type=str, help="Files to classify.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()

# Fix random seeds and number of threads
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Create the model, using mobilenet_v2 trained network.
inputs = tf.keras.layers.Input([224, 224, 3])
mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2", output_shape=[1001])
predictions = mobilenet(inputs)
model = tf.keras.Model(inputs=inputs, outputs=predictions)

for image_name in args.images:
    # Load the file
    with open(image_name, "rb") as image_file:
        image = tf.image.decode_image(image_file.read(), channels=3, dtype=tf.float32)
    # Resize to 224,224
    image = tf.image.resize(image, size=(224, 224))
    # Compute the prediction
    prediction = model.predict(tf.expand_dims(image, 0))[0]
    print("Image {}: label {}".format(image_name, imagenet_classes.imagenet_classes[np.argmax(prediction)]))
