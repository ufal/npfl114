#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import keras_cv
import numpy as np
import tensorflow as tf
tf.get_logger().addFilter(lambda m: "Analyzer.lamba_check" not in m.getMessage())  # Avoid pesky warning

parser = argparse.ArgumentParser()
parser.add_argument("prompt", type=str, help="Prompt for the model to generate.")
parser.add_argument("--images", default=3, type=int, help="Number of images to generate.")
parser.add_argument("--output", default="image", type=str, help="Filename to save the images to.")
parser.add_argument("--version", default=2, type=int, choices=[1, 2], help="Stable Diffusion version.")
args = parser.parse_args()

if args.version == 1:
    model = keras_cv.models.StableDiffusion
elif args.version == 2:
    model = keras_cv.models.StableDiffusionV2
model = model(img_height=512, img_width=512)

images = model.text_to_image(args.prompt, batch_size=args.images)

for i, image in enumerate(images):
    with open("{}-{}.jpg".format(args.output, i + 1), "wb") as jpg_file:
        jpg_file.write(tf.image.encode_jpeg(image).numpy())
