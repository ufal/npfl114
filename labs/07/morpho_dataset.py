import os
import sys
import urllib.request
import zipfile

import numpy as np

# Loads a morphological dataset in a vertical format.
# - The data consists of three Datasets
#   - train
#   - dev
#   - test
# - Each dataset is composed of factors (FORMS, LEMMAS, TAGS), each an
#   object containing the following fields:
#   - word_strings: Strings of the original words.
#   - word_ids: Word ids of the original words (uses <unk> and <pad>).
#   - words_map: String -> word_id map.
#   - words: Word_id -> string list.
#   - alphabet_map: Character -> char_id map.
#   - alphabet: Char_id -> character list.
#   - charseqs: Sequences of characters of the original words.
class MorphoDataset:
    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/datasets/"

    class Factor:
        PAD = 0
        UNK = 1
        BOW = 2
        EOW = 3

        def __init__(self, characters, train=None):
            self.words_map = train.words_map if train else {"<pad>": self.PAD, "<unk>": self.UNK}
            self.words = train.words if train else ["<pad>", "<unk>"]
            self.word_ids = []
            self.word_strings = []
            self.characters = characters
            if characters:
                self.alphabet_map = train.alphabet_map if train else {
                    "<pad>": self.PAD, "<unk>": self.UNK, "<bow>": self.BOW, "<eow>": self.EOW}
                self.alphabet = train.alphabet if train else ["<pad>", "<unk>", "<bow>", "<eow>"]
                self.charseqs = []

    class FactorBatch:
        def __init__(self, word_ids, charseqs=None):
            self.word_ids = word_ids
            self.charseqs = charseqs

    class Dataset:
        FORMS = 0
        LEMMAS = 1
        TAGS = 2
        FACTORS = 3

        def __init__(self, data_file, train=None, shuffle_batches=True, add_bow_eow=False, max_sentences=None, seed=42):
            # Create factors
            self._data = []
            for f in range(self.FACTORS):
                self._data.append(MorphoDataset.Factor(f in [self.FORMS, self.LEMMAS], train._data[f] if train else None))

            in_sentence = False
            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                if line:
                    columns = line.split("\t")
                    for f in range(self.FACTORS):
                        factor = self._data[f]
                        if not in_sentence:
                            if len(factor.word_ids): factor.word_ids[-1] = np.array(factor.word_ids[-1], np.int32)
                            factor.word_ids.append([])
                            factor.word_strings.append([])
                            if factor.characters:
                                factor.charseqs.append([])

                        word = columns[f]
                        factor.word_strings[-1].append(word)

                        # Character-level information
                        if factor.characters:
                            factor.charseqs[-1].append([])
                            if add_bow_eow:
                                factor.charseqs[-1][-1].append(MorphoDataset.Factor.BOW)
                            for c in word:
                                if c not in factor.alphabet_map:
                                    if train:
                                        c = "<unk>"
                                    else:
                                        factor.alphabet_map[c] = len(factor.alphabet)
                                        factor.alphabet.append(c)
                                factor.charseqs[-1][-1].append(factor.alphabet_map[c])
                            if add_bow_eow:
                                factor.charseqs[-1][-1].append(MorphoDataset.Factor.EOW)

                        # Word-level information
                        if word not in factor.words_map:
                            if train:
                                word = "<unk>"
                            else:
                                factor.words_map[word] = len(factor.words)
                                factor.words.append(word)
                        factor.word_ids[-1].append(factor.words_map[word])

                    in_sentence = True
                else:
                    in_sentence = False
                    if max_sentences is not None and len(self._data[self.FORMS].word_ids) >= max_sentences:
                        break

            self._size = len(self._data[self.FORMS].word_ids)
            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        def size(self):
            return self._size

        def batches(self, size=None):
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = []
                max_sentence_len = max(len(self._data[self.FORMS].word_ids[i]) for i in batch_perm)

                # Word-level data
                for factor in self._data:
                    batch.append(MorphoDataset.FactorBatch(np.zeros([batch_size, max_sentence_len], np.int32)))
                    for i in range(batch_size):
                        batch[-1].word_ids[i, :len(factor.word_ids[batch_perm[i]])] = factor.word_ids[batch_perm[i]]

                # Character-level data
                for f, factor in enumerate(self._data):
                    if not factor.characters: continue

                    max_charseq_len = max(len(charseq) for i in batch_perm for charseq in factor.charseqs[i])

                    batch[f].charseqs = np.zeros([batch_size, max_sentence_len, max_charseq_len], np.int32)
                    for i in range(batch_size):
                        for j, charseq in enumerate(factor.charseqs[batch_perm[i]]):
                            batch[f].charseqs[i, j, :len(charseq)] = charseq

                yield batch


    def __init__(self, dataset, add_bow_eow=False, max_sentences=None):
        path = "{}.zip".format(dataset)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(dataset_file,
                                                        train=self.train if dataset != "train" else None,
                                                        shuffle_batches=dataset == "train",
                                                        add_bow_eow=add_bow_eow,
                                                        max_sentences=max_sentences))
