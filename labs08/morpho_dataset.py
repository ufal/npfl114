from __future__ import division
from __future__ import print_function

import numpy as np

class MorphoDataset:
    """Class capable of loading morphological datasets in vertical format."""
    FORMS = 0
    LEMMAS = 1
    TAGS = 2
    FACTORS = 3

    def __init__(self, filename, add_bow_eow=False, train=None):
        """Load dataset from file in vertical format.

        Arguments:
        add_bow_eow: Whether to add BOW/EOW characters to the word characters.
        train: If given, the words and words_map from the training data will be reused.
        """

        # Create alphabet map
        self._alphabet_map = train._alphabet_map if train else {'<pad>': 0, '<unk>': 1, '<bow>': 2, '<eow>': 3}
        self._alphabet = train._alphabet if train else ['<pad>', '<unk>', '<bow>', '<eow>']

        # Create word maps
        self._data = []
        for f in range(self.FACTORS):
            self._data.append({})
            self._data[f]['words_map'] = train._data[f]['words_map'] if train else {'<pad>': 0, '<unk>': 1}
            self._data[f]['words'] = train._data[f]['words'] if train else ['<pad>', '<unk>']
            self._data[f]['word_ids'] = []
            self._data[f]['charseqs_map'] = {'<pad>': 0}
            self._data[f]['charseqs'] = [[self._alphabet_map['<pad>']]]
            self._data[f]['charseq_ids'] = []
            self._data[f]['strings'] = []

        # Load the sentences
        with open(filename, "r") as file:
            in_sentence = False
            for line in file:
                line = line.rstrip("\r\n")
                if line:
                    factors = line.split("\t")
                    for f in range(self.FACTORS):
                        if not in_sentence:
                            self._data[f]['word_ids'].append([])
                            self._data[f]['charseq_ids'].append([])
                            self._data[f]['strings'].append([])
                        word = factors[f] if f < len(factors) else '<pad>'
                        self._data[f]['strings'][-1].append(word)

                        # Character-level information
                        if word not in self._data[f]['charseqs_map']:
                            self._data[f]['charseqs_map'][word] = len(self._data[f]['charseqs'])
                            self._data[f]['charseqs'].append([])
                            if add_bow_eow:
                                self._data[f]['charseqs'][-1].append(self._alphabet_map['<bow>'])
                            for c in word:
                                if c not in self._alphabet_map:
                                    if train:
                                        c = '<unk>'
                                    else:
                                        self._alphabet_map[c] = len(self._alphabet)
                                        self._alphabet.append(c)
                                self._data[f]['charseqs'][-1].append(self._alphabet_map[c])
                            if add_bow_eow:
                                self._data[f]['charseqs'][-1].append(self._alphabet_map['<eow>'])
                        self._data[f]['charseq_ids'][-1].append(self._data[f]['charseqs_map'][word])

                        # Word-level information
                        if word not in self._data[f]['words_map']:
                            if train:
                                word = '<unk>'
                            else:
                                self._data[f]['words_map'][word] = len(self._data[f]['words'])
                                self._data[f]['words'].append(word)
                        self._data[f]['word_ids'][-1].append(self._data[f]['words_map'][word])
                    in_sentence = True
                else:
                    in_sentence = False

        # Compute sentence lengths
        sentences = len(self._data[0]['word_ids'])
        self._sentence_lens = np.zeros([sentences], np.int32)
        for i in range(len(self._data[0]['word_ids'])):
            self._sentence_lens[i] = len(self._data[0]['word_ids'][i])

        self._permutation = np.random.permutation(len(self._sentence_lens))

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def sentence_lens(self):
        return self._sentence_lens

    @property
    def factors(self):
        """Return the factors of the dataset.

        The result is an array of factors, each a dictionary containing:
        strings: Strings of the original words.
        word_ids: Word ids of the original words (uses <unk> and <pad>).
        words_map: String -> word_id map.
        words: Word_id -> string map.
        charseq_ids: Character_sequence ids of the original words.
        charseqs_map: String -> character_sequence_id map.
        charseqs: Character_sequence_id -> [characters], where character is an index
          to the dataset alphabet.
        """

        return self._data

    def next_batch(self, batch_size, including_charseqs=False):
        """Return the next batch.

        Arguments:
        including_charseqs: if True, also batch_charseq_ids, batch_charseqs and batch_charseq_lens are returned

        Returns: (sentence_lens, batch_word_ids[, batch_charseq_ids, batch_charseqs])
        sequence_lens: batch of sentence_lens
        batch_word_ids: for each factor, batch of words_id
        batch_charseq_ids: For each factor, batch of charseq_ids
          (the same shape as words_id, but with the ids pointing into batch_charseqs).
          Returned only if including_charseqs is True.
        batch_charseqs: For each factor, all unique charseqs in the batch,
          indexable by batch_charseq_ids. Contains indices of characters from self.alphabet.
          Returned only if including_charseqs is True.
        batch_charseq_lens: For each factor, length of charseqs in batch_charseqs.
          Returned only if including_charseqs is True.
        """

        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]
        return self._next_batch(batch_perm, including_charseqs)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentence_lens))
            return True
        return False

    def whole_data_as_batch(self, including_charseqs=False):
        """Return the whole dataset in the same result as next_batch.

        Arguments:
        including_charseqs: if True, also batch_charseq_ids, batch_charseqs and batch_charseq_lens are returned

        Returns the same results as next_batch.
        """
        return self._next_batch(np.arange(len(self.sentence_lens)), including_charseqs)

    def _next_batch(self, batch_perm, including_charseqs):
        batch_size = len(batch_perm)

        # General data
        batch_sentence_lens = self._sentence_lens[batch_perm]
        max_sentence_len = np.max(batch_sentence_lens)

        # Word-level data
        batch_word_ids = []
        for f in range(self.FACTORS):
            batch_word_ids.append(np.zeros([batch_size, max_sentence_len], np.int32))
            for i in range(batch_size):
                batch_word_ids[-1][i, 0:batch_sentence_lens[i]] = self._data[f]['word_ids'][batch_perm[i]]

        if not including_charseqs:
            return self._sentence_lens[batch_perm], batch_word_ids

        # Character-level data
        batch_charseq_ids, batch_charseqs, batch_charseq_lens = [], [], []
        for f in range(self.FACTORS):
            batch_charseq_ids.append(np.zeros([batch_size, max_sentence_len], np.int32))
            charseqs_map = {}
            charseqs = []
            charseq_lens = []
            for i in range(batch_size):
                for j, charseq_id in enumerate(self._data[f]['charseq_ids'][batch_perm[i]]):
                    if charseq_id not in charseqs_map:
                        charseqs_map[charseq_id] = len(charseqs)
                        charseqs.append(self._data[f]['charseqs'][charseq_id])
                    batch_charseq_ids[-1][i, j] = charseqs_map[charseq_id]

            batch_charseq_lens.append(np.array([len(charseq) for charseq in charseqs], np.int32))
            batch_charseqs.append(np.zeros([len(charseqs), np.max(batch_charseq_lens[-1])], np.int32))
            for i in range(len(charseqs)):
                batch_charseqs[-1][i, 0:len(charseqs[i])] = charseqs[i]

        return self._sentence_lens[batch_perm], batch_word_ids, batch_charseq_ids, batch_charseqs, batch_charseq_lens

