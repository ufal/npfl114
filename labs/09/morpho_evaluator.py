#!/usr/bin/env python3
import argparse

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=str, help="Path to predicted output.")
    parser.add_argument("gold", type=str, help="Path to gold output (extract from .zip).")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    with open(args.predictions, "r", encoding="utf-8") as predictions_file:
        predictions = [line.rstrip("\n") for line in predictions_file]

    with open(args.gold, "r", encoding="utf-8") as gold_file:
        gold = [line.rstrip("\n") for line in gold_file]

    if len(predictions) < len(gold):
        raise RuntimeError("The predictions are shorter than gold data: {} vs {}.".format(len(predictions), len(gold)))

    words, correct_lemmas, correct_tags = 0, 0, 0
    for i in range(len(gold)):
        if not gold[i]: continue

        _, gold_lemma, gold_tag = gold[i].split("\t")
        _, predicted_lemma, predicted_tag = predictions[i].split("\t")

        words += 1
        correct_lemmas += gold_lemma == predicted_lemma
        correct_tags += gold_tag == predicted_tag

    print("Lemma accuracy: {:.2f}%".format(100 * correct_lemmas / words))
    print("Tag accuracy: {:.2f}%".format(100 * correct_tags / words))
