#!/usr/bin/env python3
#bfc95faa-444e-11e9-b0fd-00505601122b
#3da961ed-4364-11e9-b0fd-00505601122b

import numpy as np

if __name__ == "__main__":
    # Load data distribution, each data point on a line

    data_counts = {}
    seq_len = 0
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            seq_len += 1
            line = line.rstrip("\n")
            if line not in data_counts.keys():
                data_counts[line] = 1
            else:
                data_counts[line] += 1

    data_probs = {}
    for k in data_counts.keys():
        data_probs[k] = data_counts[k] / seq_len

    # Load model distribution, each line `word \t probability`.
    model_probs = {}
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            split = line.split("\t")
            model_probs[split[0]] = float(split[1])

    for k in data_probs.keys():
        if k not in model_probs.keys():
            model_probs[k] = 0.0

    for k in model_probs.keys():
        if k not in data_probs.keys():
            data_probs[k] = 0.0

    data_d = []
    model_d = []

    for k in data_probs.keys():
        data_d.append(data_probs[k])
        model_d.append(model_probs[k])

    data_dist = np.array(data_d)
    model_dist = np.array(model_d)

    # Data entropy
    entropy = -np.sum(data_dist[data_dist > 0.0] * np.log(data_dist[data_dist > 0.0]))
    if not np.isnan(entropy):
        print("{:.2f}".format(entropy))
    else:
        print("{:.2f}".format(0.0))

    #Data and model cross-entropy
    cross_entropy = -np.sum(data_dist * np.log(model_dist))
    if not np.isnan(cross_entropy):
        print("{:.2f}".format(cross_entropy))
    else:
        print("{:.2f}".format(0.0))

    #KL Divergence
    D_KL = cross_entropy - entropy

    if not np.isnan(D_KL):
        print("{:.2f}".format(D_KL))
    else:
        print("{:.2f}".format(0.0))
