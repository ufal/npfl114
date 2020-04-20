import os
import sys
import pickle

import numpy as np

class TimitMFCC:
    LETTERS = [
        "<pad>", "_", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    ]

    MFCC_DIM = 26

    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42):
            self._data = {}
            self._data["mfcc"] = data["mfcc"]
            self._data["letters"] = [letters.astype(np.int32) + 1 for letters in data["letters"]]
            self._size = len(self._data["mfcc"])

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None):
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {}
                for key, values in self._data.items():
                    max_length = max(len(values[i]) for i in batch_perm)
                    batch[key] = np.zeros([batch_size, max_length, *values[batch_perm[0]].shape[1:]], values[batch_perm[0]].dtype)
                    batch[key + "_len"] = np.zeros([batch_size], dtype=np.int32)

                    for i, index in enumerate(batch_perm):
                        batch[key][i][:len(values[index])] = values[index]
                        batch[key + "_len"][i] = len(values[index])
                yield batch

    def __init__(self, path="timit_mfcc.pickle"):
        if not os.path.exists(path):
            print("The Timit dataset is not public, you need to manually download\n" +
                  "timit_mfcc.pickle file from ReCodEx.", file=sys.stderr)
            sys.exit(1)

        with open(path, "rb") as timit_mfcc_file:
            data = pickle.load(timit_mfcc_file)

        for dataset in ["train", "dev", "test"]:
            setattr(self, dataset, self.Dataset(data[dataset], shuffle_batches=dataset == "train"))
