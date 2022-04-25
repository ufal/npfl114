from __future__ import annotations
import os
import sys
from typing import Any, BinaryIO, Callable, Dict, Optional, Sequence, TextIO
import urllib.request
import zipfile
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import tensorflow as tf


# Loads a text classification dataset in a vertical format.
#
# During the construction a `tokenizer` callable taking a string
# and returning a list/np.ndarray of integers can be given. If so,
# the result of the tokenizer is available in the datasets as "tokens".
class TextClassificationDataset:
    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/datasets/"

    class Dataset:
        def __init__(
            self, data_file: BinaryIO, tokenizer: Optional[Callable[[str], Any]], train: Optional[Dataset] = None
        ) -> None:
            # Create or copy the label mapping
            if train:
                self._label_mapping = train._label_mapping
            else:
                self._label_mapping = tf.keras.layers.StringLookup(num_oov_indices=0)

            # Load the data
            self._data = {
                "documents": [],
                "labels": [],
            }
            self._data_tensors = None

            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                label, document = line.split("\t", maxsplit=1)

                self._data["documents"].append(document)
                self._data["labels"].append(label)

            self._size = len(self._data["labels"])

            # Tokenize if the tokenizer is given
            if tokenizer:
                self._data["tokens"] = [tokenizer(document) for document in self._data["documents"]]

            # Initialize the label mapping if required
            if not train:
                self._label_mapping.set_vocabulary(sorted(set(self._data["labels"])))

        @property
        def data(self) -> Dict[str, Any]:
            return self._data

        @property
        def label_mapping(self) -> tf.keras.layers.StringLookup:
            return self._label_mapping

        @property
        def size(self) -> int:
            return self._size

        @property
        def dataset(self) -> tf.data.Dataset:
            if self._data_tensors is None:
                self._data_tensors = {
                    name: (tf.ragged.constant if name == "tokens" else tf.constant)(value)
                    for name, value in self._data.items()
                }
            return tf.data.Dataset.from_tensor_slices(self._data_tensors)

    def __init__(self, name: str, tokenizer: Optional[Callable[[str], Any]] = None) -> None:
        """Create the dataset from the given filename.

        If given, the `tokenizer` should be a callable taking a string and
        returning a list/np.ndarray of integers.
        """

        path = "{}.zip".format(name)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(dataset_file, tokenizer,
                                                        train=self.train if dataset != "train" else None))

    train: Dataset
    dev: Dataset
    test: Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: Sequence[str]) -> float:
        gold = gold_dataset.data["labels"]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = sum(gold[i] == predictions[i] for i in range(len(gold)))
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        predictions = [line.rstrip("\r\n") for line in predictions_file]
        return TextClassificationDataset.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--corpus", default="czech_facebook", type=str, help="Text classification corpus")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = TextClassificationDataset.evaluate_file(
                getattr(TextClassificationDataset(args.corpus), args.dataset), predictions_file)
        print("Text classification accuracy: {:.2f}%".format(accuracy))
