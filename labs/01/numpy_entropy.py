#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: process the line

    # TODO: Create a NumPy array containing the data distribution


    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line

    # TODO: Compute and print entropy H(data distribution)
    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
