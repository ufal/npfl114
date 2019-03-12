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
    for key in data_counts.keys():
        count = data_counts[key]
        data_probs[key.lower()] = count / seq_len


    # Load model distribution, each line `word \t probability`.
    model_probs = {}
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            split = line.split("\t")
            model_probs[split[0].lower()] = (np.float(split[1]))
    
    for k in data_probs:
        if k not in model_probs.keys():
            model_probs[k] = 0
    for k in model_probs:
        if k not in data_probs.keys():
            data_probs[k] = 0
    print(data_probs)
    print(model_probs)

    data_dist = np.array([data_probs[k] for k in sorted(data_probs.keys())])
    model_dist = np.array([model_probs[k] for k in sorted(model_probs.keys())])

    # Data entropy
    tmp = data_dist[np.nonzero(data_dist)]
    entropy = -np.sum(tmp * np.log(tmp))
    print("{:.2f}".format(entropy))
    #Data and model cross-entropy
    cross_entropy = -np.sum(data_dist * np.log(model_dist))
    print("{:.2f}".format(cross_entropy))

    #KL Divergence
    D_KL = cross_entropy - entropy
    print("{:.2f}".format(D_KL))