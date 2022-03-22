import os
import sys
from typing import Dict, List, Tuple, Sequence, TextIO
import urllib.request
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf


class SVHN:
    LABELS: int = 10

    # Type alias for a bounding box -- a list of floats.
    BBox = List[float]

    # The indices of the bounding box coordinates.
    TOP: int = 0
    LEFT: int = 1
    BOTTOM: int = 2
    RIGHT: int = 3

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/datasets/"

    @staticmethod
    def parse(example: tf.Tensor) -> Dict[str, tf.Tensor]:
        example = tf.io.parse_single_example(example, {
            "image": tf.io.FixedLenFeature([], tf.string),
            "classes": tf.io.VarLenFeature(tf.int64),
            "bboxes": tf.io.VarLenFeature(tf.int64)})
        example["image"] = tf.image.decode_png(example["image"], channels=3)
        example["image"] = tf.image.convert_image_dtype(example["image"], tf.float32)
        example["classes"] = tf.sparse.to_dense(example["classes"])
        example["bboxes"] = tf.reshape(tf.cast(tf.sparse.to_dense(example["bboxes"]), tf.float32), [-1, 4])
        return example

    def __init__(self) -> None:
        for dataset, size in [("train", 10000), ("dev", 1267), ("test", 4535)]:
            path = "svhn.{}.tfrecord".format(dataset)
            if not os.path.exists(path):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

            setattr(self, dataset,
                    tf.data.TFRecordDataset(path).map(SVHN.parse).apply(tf.data.experimental.assert_cardinality(size)))

    train: tf.data.Dataset
    dev: tf.data.Dataset
    test: tf.data.Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(
        gold_dataset: tf.data.Dataset, predictions: Sequence[Tuple[List[int], List[BBox]]], iou_threshold: float = 0.5,
    ) -> float:
        def bbox_iou(x: SVHN.BBox, y: SVHN.BBox) -> float:
            def area(bbox: SVHN.BBox) -> float:
                return max(bbox[SVHN.BOTTOM] - bbox[SVHN.TOP], 0) * max(bbox[SVHN.RIGHT] - bbox[SVHN.LEFT], 0)
            intersection = [max(x[SVHN.TOP], y[SVHN.TOP]), max(x[SVHN.LEFT], y[SVHN.LEFT]),
                            min(x[SVHN.BOTTOM], y[SVHN.BOTTOM]), min(x[SVHN.RIGHT], y[SVHN.RIGHT])]
            x_area, y_area, intersection_area = area(x), area(y), area(intersection)
            return intersection_area / (x_area + y_area - intersection_area)

        gold = [(np.array(example["classes"]), np.array(example["bboxes"])) for example in gold_dataset]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = 0
        for (gold_classes, gold_bboxes), (prediction_classes, prediction_bboxes) in zip(gold, predictions):
            if len(gold_classes) != len(prediction_classes):
                continue

            used = [False] * len(gold_classes)
            for cls, bbox in zip(prediction_classes, prediction_bboxes):
                best = None
                for i in range(len(gold_classes)):
                    if used[i] or gold_classes[i] != cls:
                        continue
                    iou = bbox_iou(bbox, gold_bboxes[i])
                    if iou >= iou_threshold and (best is None or iou > best_iou):
                        best, best_iou = i, iou
                if best is None:
                    break
                used[best] = True
            correct += all(used)

        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: tf.data.Dataset, predictions_file: TextIO) -> float:
        predictions = []
        for line in predictions_file:
            values = line.split()
            if len(values) % 5:
                raise RuntimeError("Each prediction must contain multiple of 5 numbers, found {}".format(len(values)))

            predictions.append(([], []))
            for i in range(0, len(values), 5):
                predictions[-1][0].append(int(values[i]))
                predictions[-1][1].append([float(value) for value in values[i + 1:i + 5]])

        return SVHN.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = SVHN.evaluate_file(getattr(SVHN(), args.dataset), predictions_file)
        print("SVHN accuracy: {:.2f}%".format(accuracy))
