import os
import sys
from typing import Dict, Optional
import urllib.request

import tensorflow as tf


class Image64Dataset:
    H: int = 64
    W: int = 64
    C: int = 3

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2223/datasets/"

    @staticmethod
    def parse(example: tf.Tensor) -> Dict[str, tf.Tensor]:
        example = tf.io.parse_single_example(example, {
            "image": tf.io.FixedLenFeature([], tf.string),
        })
        example["image"] = tf.image.decode_png(example["image"], channels=3)
        return example

    def __init__(self, name: str, compute_size: bool = True) -> None:
        path = "{}.tfrecord".format(name)
        if not os.path.exists(path):
            print("Downloading file {}...".format(path), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}.LICENSE".format(self._URL, path), filename="{}.LICENSE".format(path))
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        self.train = tf.data.TFRecordDataset(path)
        if compute_size:
            self.train = self.train.apply(tf.data.experimental.assert_cardinality(sum(1 for _ in self.train)))
        self.train = self.train.map(Image64Dataset.parse)

    train: tf.data.Dataset
