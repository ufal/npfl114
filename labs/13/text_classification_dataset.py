import os
import sys
import urllib.request
import zipfile

import numpy as np

# Loads a text classification dataset in a vertical format.
#
# During the construction a `tokenizer` callable taking a string
# and returning a list/np.ndarray of integers must be given.
class TextClassificationDataset:
    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/datasets/"

    class Dataset:
        LABELS = None # Will be filled during Dataset construction

        def __init__(self, data_file, tokenizer, train=None, shuffle_batches=True, seed=42):
            # Create factors
            self._data = {
                "tokens": [],
                "labels": [],
            }
            self._label_map = train._label_map if train else {}
            self.LABELS = train.LABELS if train else []

            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                label, text = line.split("\t", maxsplit=1)

                if not train and label not in self._label_map:
                    self._label_map[label] = len(self._label_map)
                    self.LABELS.append(label)
                label = self._label_map.get(label, -1)

                self._data["tokens"].append(tokenizer(text))
                self._data["labels"].append(label)

            self._size = len(self._data["tokens"])
            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        def size(self):
            return self._size

        def batches(self, size=None):
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            data_tokens = self._data["tokens"]
            data_labels = self._data["labels"]

            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                max_sentence_len = max(len(data_tokens[i]) for i in batch_perm)

                tokens = np.zeros([batch_size, max_sentence_len], np.int32)
                labels = np.zeros([batch_size], np.int32)
                for i in range(batch_size):
                    tokens[i, :len(data_tokens[batch_perm[i]])] = data_tokens[batch_perm[i]]
                    labels[i] = data_labels[batch_perm[i]]

                yield tokens, labels


    def __init__(self, dataset, tokenizer):
        """Create the dataset of the given name.

        The `tokenizer` should be a callable taking a string and returning
        a list/np.ndarray of integers.
        """

        path = "{}.zip".format(dataset)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(dataset_file, tokenizer,
                                                        train=self.train if dataset != "train" else None,
                                                        shuffle_batches=dataset == "train"))
