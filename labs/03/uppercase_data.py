import os
import sys
import urllib.request
import zipfile

import numpy as np

# Loads the Uppercase data.
# - The data consists of three Datasets
#   - train
#   - dev
#   - test [all in lowercase]
# - When loading, maximum number of alphabet characters can be specified,
#   in which case that many most frequent characters will be used, and all
#   other will be remapped to "<unk>".
# - Batches are generated using a sliding window of given size,
#   i.e., for a character, we includee left `window` characters, the character
#   itself and right `window` characters, `2 * window + 1` in total.
class UppercaseData:
    LABELS = 2

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/datasets/uppercase_data.zip"

    class Dataset:
        def __init__(self, data, window, alphabet, shuffle_batches, seed=42):
            self._window = window
            self._text = data
            self._size = len(self._text)

            # Create alphabet_map
            alphabet_map = {"<pad>": 0, "<unk>": 1}
            if not isinstance(alphabet, int):
                for index, letter in enumerate(alphabet):
                    alphabet_map[letter] = index
            else:
                # Find most frequent characters
                freqs = {}
                for char in self._text.lower():
                    freqs[char] = freqs.get(char, 0) + 1

                most_frequent = sorted(freqs.items(), key=lambda item:item[1], reverse=True)
                for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                    alphabet_map[char] = i
                    if alphabet and len(alphabet_map) >= alphabet: break

            # Remap lowercased input characters uing the alphabet_map
            lcletters = np.zeros(self._size + 2 * window, np.int16)
            for i in range(self._size):
                char = self._text[i].lower()
                if char not in alphabet_map: char = "<unk>"
                lcletters[i + window] = alphabet_map[char]

            # Generate batches data
            windows = np.zeros([self._size, 2 * window + 1], np.int16)
            labels = np.zeros(self._size, np.uint8)
            for i in range(self._size):
                windows[i] = lcletters[i:i + 2 * window + 1]
                labels[i] = self._text[i].isupper()
            self._data = {"windows": windows, "labels": labels}

            # Compute alphabet
            self._alphabet = [None] * len(alphabet_map)
            for key, value in alphabet_map.items():
                self._alphabet[value] = key

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def alphabet(self):
            return self._alphabet

        @property
        def text(self):
            return self._text

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


    def __init__(self, window, alphabet_size=0):
        path = os.path.basename(self._URL)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(path), file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename=path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    data = dataset_file.read().decode("utf-8")
                setattr(self, dataset, self.Dataset(
                    data,
                    window,
                    alphabet=alphabet_size if dataset == "train" else self.train.alphabet,
                    shuffle_batches=dataset == "train",
                ))
