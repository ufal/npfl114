from __future__ import division
from __future__ import print_function

import numpy as np

class MorphoDataset:
    FORMS = 0
    LEMMAS = 1
    TAGS = 2
    FACTORS = 3

    def __init__(self, filename, add_bow_eow=False, train=None):
        # Create alphabet map
        self._alphabet_map = train._alphabet_map if train else {'<pad>': 0, '<unk>': 1, '<bow>': 2, '<eow>': 3}
        self._alphabet = train._alphabet if train else ['<pad>', '<unk>', '<bow>', '<eow>']

        # Create word maps
        self._data = []
        for f in range(self.FACTORS):
            self._data.append({})
            self._data[f]['vocabulary_map'] = train._data[f]['vocabulary_map'] if train else {'<pad>': 0, '<unk>': 1}
            self._data[f]['vocabulary'] = train._data[f]['vocabulary'] if train else ['<pad>', '<unk>']
            self._data[f]['letters'] = train._data[f]['letters'] if train else [[], []]
            self._data[f]['words'] = []

        # Load the sentences
        with open(filename, "rw") as file:
            in_sentence = False
            for line in file:
                line = line.rstrip("\r\n")
                if line:
                    factors = line.split("\t")
                    for f in range(self.FACTORS):
                        if not in_sentence:
                            self._data[f]['words'].append([])
                        word = factors[f] if f < len(factors) else '<pad>'
                        if word not in self._data[f]['vocabulary_map']:
                            if train:
                                word = '<unk>'
                            else:
                                self._data[f]['vocabulary_map'][word] = len(self._data[f]['vocabulary'])
                                self._data[f]['vocabulary'].append(word)
                                self._data[f]['letters'].append([])
                                if add_bow_eow:
                                    self._data[f]['letters'][-1].append(self._alphabet_map['<bow>'])
                                for c in word:
                                    if c not in self._alphabet_map:
                                        self._alphabet_map[c] = len(self._alphabet)
                                        self._alphabet.append(c)
                                    self._data[f]['letters'][-1].append(self._alphabet_map[c])
                                if add_bow_eow:
                                    self._data[f]['letters'][-1].append(self._alphabet_map['<eow>'])
                        self._data[f]['words'][-1].append(self._data[f]['vocabulary_map'][word])
                    in_sentence = True
                else:
                    in_sentence = False

        # Compute sentence lengths
        sentences = len(self._data[0]['words'])
        self._sentence_lens = np.zeros([sentences], np.int32)
        for i in range(len(self._data[0]['words'])):
            self._sentence_lens[i] = len(self._data[0]['words'][i])
        max_sentence_len = np.max(self._sentence_lens)

        # Fill nparrays
        for f in range(self.FACTORS):
            self._data[f]['words_tensor'] = np.zeros([sentences, max_sentence_len], np.int32)
            for i in range(len(self._data[f]['words'])):
                self._data[f]['words_tensor'][i][0:len(self._data[f]['words'][i])] = self._data[f]['words'][i]

        self._permutation = np.random.permutation(len(self._sentence_lens))

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def sentence_lens(self):
        return self._sentence_lens

    @property
    def factors(self):
        return self._data

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]
        batch_len = np.max(self._sentence_lens[batch_perm])
        batch_factors = []
        for f in range(self.FACTORS):
            batch_factors.append(self._data[f]['words_tensor'][batch_perm, 0:batch_len])
        return self._sentence_lens[batch_perm], batch_factors

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentence_lens))
            return True
        return False
