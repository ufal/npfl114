#!/usr/bin/env python3
import argparse

from omr_dataset import OMRDataset

def edit_distance(x, y):
    a = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(len(x) + 1): a[i][0] = i
    for j in range(len(y) + 1): a[0][j] = j
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            a[i][j] = min(
                a[i][j - 1] + 1,
                a[i - 1][j] + 1,
                a[i - 1][j - 1] + (x[i - 1] != y[j - 1])
            )
    return a[-1][-1]

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=str, help="Path to predicted output.")
    parser.add_argument("dataset", type=str, help="Which dataset to evaluate ('dev', 'test').")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    gold = [marks.numpy() for marks in getattr(OMRDataset(), args.dataset).map(OMRDataset.parse).map(lambda example: example["marks"])]

    with open(args.predictions, "r", encoding="utf-8") as predictions_file:
        predictions = [line.rstrip("\n") for line in predictions_file]

    if len(predictions) < len(gold):
        raise RuntimeError("The predictions are shorter than gold data: {} vs {}.".format(len(predictions), len(gold)))

    score = 0
    for i in range(len(gold)):
        gold_sentence = [OMRDataset.MARKS[mark] for mark in gold[i]]
        predicted_sentence = predictions[i].split(" ")
        score += edit_distance(gold_sentence, predicted_sentence) / len(gold_sentence)

    print("Average normalized edit distance: {:.3f}%".format(100 * score / len(gold)))
