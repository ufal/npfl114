import os
import sys
import urllib.request
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

class CAGS:
    H, W, C = 224, 224, 3
    LABELS = [
        # Cats
        "Abyssinian", "Bengal", "Bombay", "British_Shorthair", "Egyptian_Mau",
        "Maine_Coon", "Russian_Blue", "Siamese", "Sphynx",
        # Dogs
        "american_bulldog", "american_pit_bull_terrier", "basset_hound",
        "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
        "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
        "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
        "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
        "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
        "wheaten_terrier", "yorkshire_terrier",
    ]

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/datasets/"

    @staticmethod
    def parse(example):
        example = tf.io.parse_single_example(example, {
            "image": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64)})
        example["image"] = tf.image.convert_image_dtype(tf.image.decode_jpeg(example["image"], channels=3), tf.float32)
        example["mask"] = tf.image.convert_image_dtype(tf.image.decode_png(example["mask"], channels=1), tf.float32)
        return example

    def __init__(self):
        for dataset in ["train", "dev", "test"]:
            path = "cags.{}.tfrecord".format(dataset)
            if not os.path.exists(path):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

            setattr(self, dataset, tf.data.TFRecordDataset(path).map(CAGS.parse))

    # Keras IoU metric
    class MaskIoUMetric(tf.metrics.Mean):
        """MaskIoUMetric computes IoU for CAGS dataset masks predicted by binary classification"""

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true_mask = tf.reshape(tf.math.round(y_true) == 1, [-1, CAGS.H * CAGS.W])
            y_pred_mask = tf.reshape(tf.math.round(y_pred) == 1, [-1, CAGS.H * CAGS.W])

            intersection_mask = tf.math.logical_and(y_true_mask, y_pred_mask)
            union_mask = tf.math.logical_or(y_true_mask, y_pred_mask)

            intersection = tf.reduce_sum(tf.cast(intersection_mask, tf.float32), axis=1)
            union = tf.reduce_sum(tf.cast(union_mask, tf.float32), axis=1)

            iou = tf.where(union == 0, 1., intersection / union)
            return super().update_state(iou, sample_weight)

    # Evaluation infrastructure.
    @staticmethod
    def evaluate_classification(gold_dataset, predictions):
        gold = [int(example["label"]) for example in gold_dataset]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = sum(gold[i] == predictions[i] for i in range(len(gold)))
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_classification_file(gold_dataset, predictions_file):
        predictions = [int(line) for line in predictions_file]
        return CAGS.evaluate_classification(gold_dataset, predictions)

    @staticmethod
    def evaluate_segmentation(gold_dataset, predictions):
        gold = [example["mask"] for example in gold_dataset]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        iou = CAGS.MaskIoUMetric()
        for i in range(len(gold)):
            iou(gold[i], predictions[i])

        return 100 * iou.result()

    @staticmethod
    def evaluate_segmentation_file(gold_dataset, predictions_file):
        predictions = []
        for line in predictions_file:
            runs = [int(run) for run in line.split()]
            assert sum(runs) == CAGS.H * CAGS.W

            offset = 0
            predictions.append(np.zeros([CAGS.H * CAGS.W], np.int32))
            for i, run in enumerate(runs):
                predictions[-1][offset:offset + run] = i % 2
                offset += run

        return CAGS.evaluate_segmentation(gold_dataset, predictions)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    parser.add_argument("--task", default="classification", type=str, help="Task to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        if args.task == "classification":
            with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
                accuracy = CAGS.evaluate_classification_file(getattr(CAGS(), args.dataset), predictions_file)
            print("CAGS accuracy: {:.2f}%".format(accuracy))
        if args.task == "segmentation":
            with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
                iou = CAGS.evaluate_segmentation_file(getattr(CAGS(), args.dataset), predictions_file)
            print("CAGS IoU: {:.2f}%".format(iou))
