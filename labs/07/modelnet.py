import os
import sys
import urllib.request

import numpy as np

class ModelNet:
    # The D, H, W are set in the constructor depending
    # on requested resolution and are only instance variables.
    D, H, W, C = None, None, None, 1
    LABELS = [
        "bathtub", "bed", "chair", "desk", "dresser",
        "monitor", "night_stand", "sofa", "table", "toilet",
    ]

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/datasets/modelnet{}.npz"

    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42):
            self._data = data
            self._size = len(self._data["voxels"])

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
                for key in self._data:
                    batch[key] = self._data[key][batch_perm]
                yield batch

    # The resolution parameter can be either 20 or 32.
    def __init__(self, resolution):
        assert resolution in [20, 32], "Only 20 or 32 resolution is supported"

        self.D = self.H = self.W = resolution
        url = self._URL.format(resolution)

        path = os.path.basename(url)
        if not os.path.exists(path):
            print("Downloading {} dataset...".format(path), file=sys.stderr)
            urllib.request.urlretrieve(url, filename=path)

        mnist = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = dict((key[len(dataset) + 1:], mnist[key]) for key in mnist if key.startswith(dataset))
            setattr(self, dataset, self.Dataset(data, shuffle_batches=dataset == "train"))
