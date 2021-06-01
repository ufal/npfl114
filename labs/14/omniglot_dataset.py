import os
import sys
import urllib.request

import numpy as np

class Omniglot:
    H, W, C = 28, 28, 1

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/datasets/"

    class Dataset:
        def __init__(self, data):
            self._data = data
            self._data["images"] = self._data["images"].astype(np.float32) / 255
            self._size = len(self._data["images"])

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        @property
        def dataset(self):
            import tensorflow as tf
            return tf.data.Dataset.from_tensor_slices(self._data)

    def __init__(self, dataset="omniglot"):
        path = "{}.npz".format(dataset)
        if not os.path.exists(path):
            print("Downloading Omniglot dataset...", file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

        omniglot = np.load(path)
        for dataset in ["train", "test"]:
            data = dict((key[len(dataset) + 1:], omniglot[key]) for key in omniglot if key.startswith(dataset))
            setattr(self, dataset, self.Dataset(data))
