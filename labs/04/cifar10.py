import os
import sys
import urllib.request

import numpy as np

class CIFAR10:
    H, W, C = 32, 32, 3
    LABELS = 10

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/datasets/cifar10_competition.npz"

    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42):
            self._data = data
            self._data["images"] = self._data["images"].astype(np.float32) / 255
            self._data["labels"] = self._data["labels"].ravel()
            self._size = len(self._data["images"])

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

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

    def __init__(self, size={}):
        path = os.path.basename(self._URL)
        if not os.path.exists(path):
            print("Downloading CIFAR-10 dataset...", file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename=path)

        cifar = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = dict((key[len(dataset) + 1:], cifar[key][:size.get(dataset, None)]) for key in cifar if key.startswith(dataset))
            setattr(self, dataset, self.Dataset(data, shuffle_batches=dataset == "train"))

    @staticmethod
    def evaluate(gold_dataset, predictions):
        gold = gold_dataset.data["labels"]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = sum(gold[i] == predictions[i] for i in range(len(gold)))
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset, predictions_file):
        predictions = [int(line) for line in predictions_file]
        return CIFAR10.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = CIFAR10.evaluate_file(getattr(CIFAR10(), args.dataset), predictions_file)
        print("CIFAR10 accuracy: {:.2f}%".format(accuracy))
