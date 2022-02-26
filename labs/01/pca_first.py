#!/usr/bin/env python3
import argparse
import os
from typing import Tuple
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--examples", default=256, type=int, help="MNIST examples to use.")
parser.add_argument("--iterations", default=100, type=int, help="Iterations of the power algorithm.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> Tuple[float, float]:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load data
    mnist = MNIST()

    data_indices = np.random.choice(mnist.train.size, size=args.examples, replace=False)
    data = tf.convert_to_tensor(mnist.train.data["images"][data_indices])

    # TODO: Data has shape [args.examples, MNIST.H, MNIST.W, MNIST.C].
    # We want to reshape it to [args.examples, MNIST.H * MNIST.W * MNIST.C].
    # We can do so using `tf.reshape(data, new_shape)` with new shape
    # `[data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]]`.
    data = None

    # TODO: Now compute mean of every feature. Use `tf.math.reduce_mean`,
    # and set `axis` to zero -- therefore, the mean will be computed
    # across the first dimension, so across examples.
    mean = None

    # TODO: Compute the covariance matrix. The covariance matrix is
    #   (data - mean)^T * (data - mean) / data.shape[0]
    # where transpose can be computed using `tf.transpose` and matrix
    # multiplication using either Python operator @ or `tf.linalg.matmul`.
    cov = None

    # TODO: Compute the total variance, which is sum of the diagonal
    # of the covariance matrix. To extract the diagonal use `tf.linalg.diag_part`
    # and to sum a tensor use `tf.math.reduce_sum`.
    total_variance = None

    # TODO: Now run `args.iterations` of the power iteration algorithm.
    # Start with a vector of `cov.shape[0]` ones of type tf.float32 using `tf.ones`.
    v = None
    for i in range(args.iterations):
        # TODO: In the power iteration algorithm, we compute
        # 1. v = cov * v
        #    The matrix-vector multiplication can be computed using `tf.linalg.matvec`.
        # 2. s = l2_norm(v)
        #    The l2_norm can be computed using `tf.norm`.
        # 3. v = v / s
        pass

    # The `v` is now the eigenvector of the largest eigenvalue, `s`. We now
    # compute the explained variance, which is a ration of `s` and `total_variance`.
    explained_variance = s / total_variance

    # Return the total and explained variance for ReCodEx to validate
    return total_variance, 100 * explained_variance

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    total_variance, explained_variance = main(args)
    print("Total variance: {:.2f}".format(total_variance))
    print("Explained variance: {:.2f}%".format(explained_variance))
