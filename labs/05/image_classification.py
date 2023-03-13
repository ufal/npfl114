#!/usr/bin/env python3
import argparse
import os
import time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+", type=str, help="Files to classify.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load EfficientNetV2-B0
    efficientnetv2_b0 = tf.keras.applications.EfficientNetV2B0(include_top=True)

    for image_path in args.images:
        # Load the file
        with open(image_path, "rb") as image_file:
            image = tf.image.decode_image(image_file.read(), channels=3, dtype=tf.uint8)

        # Resize to 224,224
        image = tf.image.resize(image, size=(224, 224))
        image = tf.keras.applications.efficientnet_v2.preprocess_input(image)

        # Compute the prediction
        start = time.time()

        predictions = efficientnetv2_b0.predict(tf.expand_dims(image, 0))

        predictions = tf.keras.applications.efficientnet_v2.decode_predictions(predictions)

        print("Image {} [{} ms] labels:{}".format(
            image_path,
            1000 * (time.time() - start),
            "".join("\n- {}: {}".format(label, prob) for _, label, prob in predictions[0])
        ))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
