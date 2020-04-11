import os
import sys
import urllib.request

import tensorflow as tf

class SVHN:
    TOP, LEFT, BOTTOM, RIGHT = range(4)
    LABELS = 10

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/datasets/"

    @staticmethod
    def parse(example):
        example = tf.io.parse_single_example(example, {
            "image": tf.io.FixedLenFeature([], tf.string),
            "classes": tf.io.VarLenFeature(tf.int64),
            "bboxes": tf.io.VarLenFeature(tf.int64)})
        example["image"] = tf.image.decode_png(example["image"], channels=3)
        example["image"] = tf.image.convert_image_dtype(example["image"], tf.float32)
        example["classes"] = tf.sparse.to_dense(example["classes"])
        example["bboxes"] = tf.reshape(tf.sparse.to_dense(example["bboxes"]), [-1, 4])
        return example

    def __init__(self):
        for dataset in ["train", "dev", "test"]:
            path = "svhn.{}.tfrecord".format(dataset)
            if not os.path.exists(path):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

            setattr(self, dataset, tf.data.TFRecordDataset(path))
