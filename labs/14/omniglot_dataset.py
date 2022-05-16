import os
import sys
from typing import Dict, Iterator, Optional
import urllib.request

import numpy as np
import tensorflow as tf


class Omniglot:
    H: int = 28
    W: int = 28
    C: int = 1

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/datasets/"

    class Dataset:
        def __init__(self, data: Dict[str, np.ndarray]) -> None:
            self._data = data
            self._data["images"] = self._data["images"].astype(np.float32) / 255
            self._size = len(self._data["images"])

        @property
        def data(self) -> Dict[str, np.ndarray]:
            return self._data

        @property
        def size(self) -> int:
            return self._size

        @property
        def dataset(self) -> tf.data.Dataset:
            return tf.data.Dataset.from_tensor_slices(self._data)

    def __init__(self, dataset: str = "omniglot"):
        path = "{}.npz".format(dataset)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

        omniglot = np.load(path)
        for dataset in ["train", "test"]:
            data = {key[len(dataset) + 1:]: omniglot[key] for key in omniglot if key.startswith(dataset)}
            setattr(self, dataset, self.Dataset(data))

    train: Dataset
    test: Dataset
