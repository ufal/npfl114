from __future__ import division
from __future__ import print_function

import numpy as np

class WordEmbeddings:
    def __init__(self, filename):
        # Load the word embeddings
        with open(filename, "rw") as file:
            line = file.readline()
            words, dimension = map(int, line.split(" "))

            self._words = []
            self._words_map = {}
            self._dimension = dimension
            self._we = np.zeros([words, dimension], np.float32)

            for i in range(words):
                line = file.readline().rstrip("\r\n")
                if not line:
                    raise ValueError("The word embedding file {} is too short, it should have contained {} words!".format(filename, words))
                parts = line.split(" ")
                self._words_map[parts[0]] = len(self._words)
                self._words.append(parts[0])
                self._we[i] = map(float, parts[1:])

    @property
    def words(self):
        return self._words

    @property
    def words_map(self):
        return self._words_map

    @property
    def dimension(self):
        return self._dimension

    @property
    def we(self):
        return self._we
