import os
import sys
import urllib.request
import zipfile

import tensorflow as tf

# Loads a morphological dataset in a vertical format.
# - The data consists of three Datasets
#   - train
#   - dev
#   - test
# - Each dataset is composed of
#   - size: number of sentences in the dataset
#   - forms, lemmas, tags: objects containing the following fields:
#     - strings: a Python list containing input sentences, each being
#         a list of strings (forms/lemmas/tags)
#     - word_mapping: a tf.keras.layers.experimental.preprocessing.StringLookup
#         object capable of mapping words to indices. It is constructed on
#         the train set and shared by the dev and test sets.
#     - char_mapping: a tf.keras.layers.experimental.preprocessing.StringLookup
#         object capable of mapping characters to indices. It is constructed on
#         the train set and shared by the dev and test sets.
#   - dataset: a tf.data.Dataset containing the (forms, lemmas, tags) triples.
class MorphoDataset:
    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/datasets/"

    class Factor:
        def __init__(self):
            self.strings = []

        def finalize(self, train=None):
            # Create mappings
            if train:
                self.word_mapping = train.word_mapping
                self.char_mapping = train.char_mapping
            else:
                self.word_mapping = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
                self.word_mapping.adapt(sorted(set(string for sentence in self.strings for string in sentence)))

                self.char_mapping = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
                self.char_mapping.adapt(sorted(set(char for sentence in self.strings for string in sentence for char in string)))

    class Dataset:
        def __init__(self, data_file, train=None, max_sentences=None):
            # Create factors
            self.forms = MorphoDataset.Factor()
            self.lemmas = MorphoDataset.Factor()
            self.tags = MorphoDataset.Factor()
            factors = [self.forms, self.lemmas, self.tags]

            # Load the data
            self.size = 0
            in_sentence = False
            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                if line:
                    if not in_sentence:
                        for factor in factors:
                            factor.strings.append([])
                        self.size += 1

                    columns = line.split("\t")
                    assert len(columns) == len(factors)
                    for column, factor in zip(columns, factors):
                        factor.strings[-1].append(column)

                    in_sentence = True
                else:
                    in_sentence = False
                    if max_sentences is not None and self.size >= max_sentences:
                        break

            # Finalize the mappings
            for factor, train_factor in zip(factors, [train.forms, train.lemmas, train.tags] if train else [None] * 3):
                factor.finalize(train_factor)

            # Create the dataset
            self.dataset = tf.data.Dataset.from_generator(
                lambda: (tuple(factor.strings[i] for factor in factors) for i in range(self.size)),
                output_signature=(tf.TensorSpec([None], tf.string),) * 3,
            ).cache()

    def __init__(self, dataset, max_sentences=None):

        path = "{}.zip".format(dataset)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(dataset_file,
                                                        train=self.train if dataset != "train" else None,
                                                        max_sentences=max_sentences))
