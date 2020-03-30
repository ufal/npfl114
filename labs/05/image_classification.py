#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import tensorflow as tf

import efficient_net
import imagenet_classes

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=str, help="Files to classify.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load EfficientNet=B0
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=True)

    for image_path in args.images:
        # Load the file
        with open(image_path, "rb") as image_file:
            image = tf.image.decode_image(image_file.read(), channels=3, dtype=tf.float32)

        # Resize to 224,224
        image = tf.image.resize(image, size=(224, 224))

        # Compute the prediction
        start = time.time()
        [prediction], *_ = efficientnet_b0.predict(tf.expand_dims(image, 0))
        print("Image {} [{} ms]: label {}".format(
            image_path,
            1000 * (time.time() - start),
            imagenet_classes.imagenet_classes[tf.argmax(prediction)]
        ))

