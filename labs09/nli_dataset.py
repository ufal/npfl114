from __future__ import division
from __future__ import print_function

import numpy as np

class NLIDataset:
    """Class capable of loading NLI dataset."""

    def __init__(self, filename, add_bow_eow=False, train=None, no_languages=False):
        """Load dataset from a given file.

        Arguments:
        add_bow_eow: Whether to add BOW/EOW characters to the word characters.
        train: If given, the vocabularies from the training data will be reused.
        """

        # Create vocabulary_maps
        if train:
            self._vocabulary_maps = train._vocabulary_maps
        else:
            self._vocabulary_maps = {'chars': {'<pad>': 0, '<unk>': 1, '<bow>': 2, '<eow>': 3},
                                     'words': {'<pad>': 0, '<unk>': 1, '\n': 2}, # \n represents EOS
                                     'tags': {'<pad>': 0, '<unk>': 1, '\n': 2}, # \n represents EOS
                                     'languages': {},
                                     'levels': {},
                                     'prompts': {}}
        self._word_ids = []
        self._charseq_ids = []
        self._charseqs_map = {'<pad>': 0}
        self._charseqs = []
        self._tags = []
        self._languages = []
        self._levels = []
        self._prompts = []

        # Load the sentences
        with open(filename, "r") as file:
            for line in file:
                line = line.rstrip("\r\n")
                language, prompt, level, words = line.split("\t", 3)
                if not train:
                    if language not in self._vocabulary_maps['languages']:
                        self._vocabulary_maps['languages'][language] = len(self._vocabulary_maps['languages'])
                    if level not in self._vocabulary_maps['levels']:
                        self._vocabulary_maps['levels'][level] = len(self._vocabulary_maps['levels'])
                    if prompt not in self._vocabulary_maps['prompts']:
                        self._vocabulary_maps['prompts'][prompt] = len(self._vocabulary_maps['prompts'])
                self._languages.append(self._vocabulary_maps['languages'][language] if not no_languages else -1)
                self._levels.append(self._vocabulary_maps['levels'][level])
                self._prompts.append(self._vocabulary_maps['prompts'][prompt])

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
                            self._charseqs[-1].append(self._vocabulary_maps['chars']['<bow>'])
                        for c in word:
                            if c not in self._vocabulary_maps['chars']:
                                if not train:
                                    self._vocabulary_maps['chars'][c] = len(self._vocabulary_maps['chars'])
                                else:
                                    c = '<unk>'
                            self._charseqs[-1].append(self._vocabulary_maps['chars'][c])
                        if add_bow_eow:
                            self._charseqs[-1].append(self._vocabulary_maps['chars']['<eow>'])
                    self._charseq_ids[-1].append(self._charseqs_map[word])

                    # Words
                    if word not in self._vocabulary_maps['words']:
                        if not train:
                            self._vocabulary_maps['words'][word] = len(self._vocabulary_maps['words'])
                        else:
                            word = '<unk>'
                    self._word_ids[-1].append(self._vocabulary_maps['words'][word])

                    # Tags
                    if tag not in self._vocabulary_maps['tags']:
                        if not train:
                            self._vocabulary_maps['tags'][tag] = len(self._vocabulary_maps['tags'])
                        else:
                            tag = '<unk>'
                    self._tags[-1].append(self._vocabulary_maps['tags'][tag])

        # Compute sentence lengths
        sentences = len(self._word_ids)
        self._sentence_lens = np.zeros([sentences], np.int32)
        for i in range(sentences):
            self._sentence_lens[i] = len(self._word_ids[i])

        # Create vocabularies
        if train:
            self._vocabularies = train._vocabularies
        else:
            self._vocabularies = {}
            for feature, words in self._vocabulary_maps.items():
                self._vocabularies[feature] = [""] * len(words)
                for word, id in words.items():
                    self._vocabularies[feature][id] = word

        self._permutation = np.random.permutation(len(self._sentence_lens))

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

    def next_batch(self, batch_size):
        """Return the next batch.

        Arguments:
        Returns: (sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages)
        sequence_lens: batch of sentence_lens
        word_ids: batch of word_ids
        charseq_ids: batch of charseq_ids (the same shape as word_ids, but with the ids pointing into charseqs).
        charseqs: unique charseqs in the batch, indexable by charseq_ids;
          contain indices of characters from vocabulary('chars')
        charseq_lens: length of charseqs
        tags: batch of tags (the same shape as word_ids)
        levels: batch of student levels
        prompts: batch of student prompts
        languages: batch of languages
        """

        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]
        return self._next_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentence_lens))
            return True
        return False

    def whole_data_as_batch(self):
        """Return the whole dataset in the same result as next_batch.

        Returns the same results as next_batch.
        """
        return self._next_batch(np.arange(len(self._sentence_lens)))

    def _next_batch(self, batch_perm):
        batch_size = len(batch_perm)

        # General data
        batch_sentence_lens = self._sentence_lens[batch_perm]
        max_sentence_len = np.max(batch_sentence_lens)

        # Word-level data
        batch_word_ids = np.zeros([batch_size, max_sentence_len], np.int32)
        for i in range(batch_size):
            batch_word_ids[i, 0:batch_sentence_lens[i]] = self._word_ids[batch_perm[i]]
        batch_tags = np.zeros([batch_size, max_sentence_len], np.int32)
        for i in range(batch_size):
            batch_tags[i, 0:batch_sentence_lens[i]] = self._tags[batch_perm[i]]
        batch_levels = np.zeros([batch_size], np.int32)
        for i in range(batch_size):
            batch_levels[i] = self._levels[batch_perm[i]]
        batch_prompts = np.zeros([batch_size], np.int32)
        for i in range(batch_size):
            batch_prompts[i] = self._prompts[batch_perm[i]]
        batch_languages = np.zeros([batch_size], np.int32)
        for i in range(batch_size):
            batch_languages[i] = self._languages[batch_perm[i]]

        # Character-level data
        batch_charseq_ids = np.zeros([batch_size, max_sentence_len], np.int32)
        charseqs_map, charseqs, charseq_lens = {}, [], []
        for i in range(batch_size):
            for j, charseq_id in enumerate(self._charseq_ids[batch_perm[i]]):
                if charseq_id not in charseqs_map:
                    charseqs_map[charseq_id] = len(charseqs)
                    charseqs.append(self._charseqs[charseq_id])
                batch_charseq_ids[i, j] = charseqs_map[charseq_id]

        batch_charseq_lens = np.array([len(charseq) for charseq in charseqs], np.int32)
        batch_charseqs = np.zeros([len(charseqs), np.max(batch_charseq_lens)], np.int32)
        for i in range(len(charseqs)):
            batch_charseqs[i, 0:len(charseqs[i])] = charseqs[i]

        return batch_sentence_lens, batch_word_ids, batch_charseq_ids, batch_charseqs, batch_charseq_lens, \
            batch_tags, batch_levels, batch_prompts, batch_languages

