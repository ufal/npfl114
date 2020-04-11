#!/usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow as tf

import bboxes_utils
from svhn_dataset import SVHN

def correct_predictions(gold_classes, gold_bboxes, predicted_classes, predicted_bboxes, iou_threshold=0.5):
    if len(gold_classes) != len(predicted_classes):
        return False

    used = [False] * len(gold_classes)
    for cls, bbox in zip(predicted_classes, predicted_bboxes):
        best = None
        for i in range(len(gold_classes)):
            if used[i] or gold_classes[i] != cls:
                continue
            iou = bboxes_utils.bbox_iou(bbox, gold_bboxes[i])
            if iou >= iou_threshold and (best is None or iou > best_iou):
                best, best_iou = i, iou
        if best is None:
            return False
        used[best] = True
    return True

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=str, help="Path to predicted output.")
    parser.add_argument("dataset", type=str, help="Which dataset to evaluate ('dev', 'test').")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load the gold data
    gold = getattr(SVHN(), args.dataset).map(SVHN.parse)

    # Read the predictions
    correct, total = 0,  0
    with open(args.predictions, "r", encoding="utf-8-sig") as predictions_file:
        for example in gold:
            predictions = [int(value) for value in predictions_file.readline().split()]
            assert len(predictions) % 5 == 0

            predictions = np.array(predictions, np.int).reshape([-1, 5])
            correct += correct_predictions(example["classes"].numpy(), example["bboxes"].numpy(),
                                           predictions[:, 0], predictions[:, 1:])
            total += 1

    print("{:.2f}".format(100 * correct / total))
