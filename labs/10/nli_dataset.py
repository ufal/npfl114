import os
import sys
import urllib.request
import zipfile

import numpy as np

class NLIDataset:
    class Batch:
        """ Batch data type.

        Each batch is an object containing
        - word_ids: batch of word_ids
        - charseq_ids: batch of charseq_ids (the same shape as word_ids, but with the ids pointing into charseqs).
        - charseqs: unique charseqs in the batch, indexable by charseq_ids; contain indices of characters from vocabulary('chars')
        - tags: batch of tags (the same shape as word_ids)
        - levels: batch of student levels
        - prompts: batch of student prompts
        - languages: batch of languages
        """
        def __init__(self, word_ids, charseq_ids, charseqs, tags, levels, prompts, languages):
            self.word_ids, self.charseq_ids, self.charseqs, self.tags, self.levels, self.prompts, self.languages = \
                word_ids, charseq_ids, charseqs, tags, levels, prompts, languages

    class Dataset:
        PAD = 0
        UNK = 1
        BOW, EOW = 2, 3
        EOS = 2

        def __init__(self, data_file, train=None, shuffle_batches=True, add_bow_eow=False, seed=42):
            # Create vocabulary_maps
            if train:
                self._vocabulary_maps = train._vocabulary_maps
            else:
                self._vocabulary_maps = {"chars": {"<pad>": self.PAD, "<unk>": self.UNK, "<bow>": self.BOW, "<eow>": self.EOW},
                                         "words": {"<pad>": self.PAD, "<unk>": self.UNK, "\n": self.EOS}, # \n represents EOS
                                         "tags": {"<pad>": self.PAD, "<unk>": self.UNK, "\n": self.EOS}, # \n represents EOS
                                         "languages": {},
                                         "levels": {},
                                         "prompts": {}}
            self._word_ids = []
            self._charseq_ids = []
            self._charseqs_map = {"<pad>": self.PAD}
            self._charseqs = [[self.PAD]]
            self._tags = []
            self._languages = []
            self._levels = []
            self._prompts = []

            # Load the sentences
            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                language, prompt, level, words = line.split("\t", 3)
                if not train:
                    if language not in self._vocabulary_maps["languages"]:
                        self._vocabulary_maps["languages"][language] = len(self._vocabulary_maps["languages"])
                    if level not in self._vocabulary_maps["levels"]:
                        self._vocabulary_maps["levels"][level] = len(self._vocabulary_maps["levels"])
                    if prompt not in self._vocabulary_maps["prompts"]:
                        self._vocabulary_maps["prompts"][prompt] = len(self._vocabulary_maps["prompts"])
                self._languages.append(self._vocabulary_maps["languages"].get(language, -1)) # Use -1 for test set
                self._levels.append(self._vocabulary_maps["levels"][level])
                self._prompts.append(self._vocabulary_maps["prompts"][prompt])

                self._word_ids.append([])
                self._tags.append([])
                self._charseq_ids.append([])
                for word_tag in words.split("\t"):
                    word, tag = word_tag.split(" ") if len(word_tag) else ("\n", "\n")

                    # Characters
                    if word not in self._charseqs_map:
                        self._charseqs_map[word] = len(self._charseqs)
                        self._charseqs.append([])
                        if add_bow_eow:
                            self._charseqs[-1].append(self.BOW)
                        for c in word:
                            if c not in self._vocabulary_maps["chars"]:
                                if not train:
                                    self._vocabulary_maps["chars"][c] = len(self._vocabulary_maps["chars"])
                                else:
                                    c = "<unk>"
                            self._charseqs[-1].append(self._vocabulary_maps["chars"][c])
                        if add_bow_eow:
                            self._charseqs[-1].append(self.EOW)
                    self._charseq_ids[-1].append(self._charseqs_map[word])

                    # Words
                    if word not in self._vocabulary_maps["words"]:
                        if not train:
                            self._vocabulary_maps["words"][word] = len(self._vocabulary_maps["words"])
                        else:
                            word = "<unk>"
                    self._word_ids[-1].append(self._vocabulary_maps["words"][word])

                    # Tags
                    if tag not in self._vocabulary_maps["tags"]:
                        if not train:
                            self._vocabulary_maps["tags"][tag] = len(self._vocabulary_maps["tags"])
                        else:
                            tag = "<unk>"
                    self._tags[-1].append(self._vocabulary_maps["tags"][tag])

            # Create vocabularies
            if train:
                self._vocabularies = train._vocabularies
            else:
                self._vocabularies = {}
                for feature, words in self._vocabulary_maps.items():
                    self._vocabularies[feature] = [""] * len(words)
                    for word, id in words.items():
                        self._vocabularies[feature][id] = word

            self._size = len(self._word_ids)
            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        def vocabulary(self, feature):
            """Return vocabulary for required feature.

            The features are the following:
            words
            chars
            tags
            languages
            levels
            prompts
            """
            return self._vocabularies[feature]

        def size(self):
            return self._size

        def batches(self, size=None):
            """ Generate the batches."""
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                max_sentence_len = max(len(self._word_ids[i]) for i in batch_perm)

                # Word-level data
                word_ids = np.zeros([batch_size, max_sentence_len], np.int32)
                for i in range(batch_size):
                    word_ids[i, 0:len(self._word_ids[batch_perm[i]])] = self._word_ids[batch_perm[i]]

                tags = np.zeros([batch_size, max_sentence_len], np.int32)
                for i in range(batch_size):
                    tags[i, 0:len(self._tags[batch_perm[i]])] = self._tags[batch_perm[i]]

                levels = np.zeros([batch_size], np.int32)
                for i in range(batch_size):
                    levels[i] = self._levels[batch_perm[i]]

                prompts = np.zeros([batch_size], np.int32)
                for i in range(batch_size):
                    prompts[i] = self._prompts[batch_perm[i]]

                languages = np.zeros([batch_size], np.int32)
                for i in range(batch_size):
                    languages[i] = self._languages[batch_perm[i]]

                # Character-level data
                charseq_ids = np.zeros([batch_size, max_sentence_len], np.int32)
                charseqs_map = {"<pad>": self.PAD}
                charseqs = [self._charseqs[self.PAD]]
                for i in range(batch_size):
                    for j, charseq_id in enumerate(self._charseq_ids[batch_perm[i]]):
                        if charseq_id not in charseqs_map:
                            charseqs_map[charseq_id] = len(charseqs)
                            charseqs.append(self._charseqs[charseq_id])
                        charseq_ids[i, j] = charseqs_map[charseq_id]

                dense_charseqs = np.zeros([len(charseqs), max(len(charseq) for charseq in charseqs)], np.int32)
                for i in range(len(charseqs)):
                    dense_charseqs[i, 0:len(charseqs[i])] = charseqs[i]

                return NLIDataset.Batch(word_ids, charseq_ids, dense_charseqs, tags, levels, prompts, languages)


    def __init__(self, path="nli_dataset.zip", add_bow_eow=False):
        if not os.path.exists(path):
            print("The NLI dataset is not public, you need to manually download\n" +
                  "nli_dataset.zip file from ReCodEx.", file=sys.stderr)
            sys.exit(1)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(dataset_file,
                                                        train=self.train if dataset != "train" else None,
                                                        shuffle_batches=dataset == "train",
                                                        add_bow_eow=add_bow_eow))
