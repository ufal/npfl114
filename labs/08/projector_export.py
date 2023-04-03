#!/usr/bin/env python
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
from tensorboard.plugins import projector
import tensorflow as tf

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_embeddings", type=str, help="Embedding file to use.")
    parser.add_argument("--elements", default=20000, type=int, help="Words to export.")
    parser.add_argument("--output_dir", default="embeddings", type=str, help="Output directory.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Generate the embeddings for the projector
    tf.summary.create_file_writer(args.output_dir)
    with open(args.input_embeddings, "r") as embedding_file:
        _, dim = map(int, embedding_file.readline().split())

        embeddings = np.zeros([args.elements, dim], np.float32)
        with open(os.path.join(args.output_dir, "metadata.tsv"), "w") as metadata_file:
            for i, line in zip(range(args.elements), embedding_file):
                form, *embedding = line.split()
                print(form, file=metadata_file)
                embeddings[i] = list(map(float, embedding))

    # Save the variable
    embeddings = tf.Variable(embeddings, tf.float32)
    checkpoint = tf.train.Checkpoint(embeddings=embeddings)
    checkpoint.save(os.path.join(args.output_dir, "embeddings.ckpt"))

    # Set up the projector config
    config = projector.ProjectorConfig()
    embeddings = config.embeddings.add()

    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embeddings.tensor_name = "embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    embeddings.metadata_path = "metadata.tsv"
    projector.visualize_embeddings(args.output_dir, config)
