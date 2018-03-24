#!/usr/bin/env python3
import argparse
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("gold", type=str, help="Path to gold data.")
parser.add_argument("system", type=str, nargs="+", help="Path[s] to system output.")
args = parser.parse_args()

gold = np.load(args.gold)
gold_labels = gold["labels"]
gold_masks = gold["masks"]

for system_path in args.system:
    with open(system_path, "r", encoding="utf-8") as system_file:
        system = system_file.readlines()

    if len(system) != len(gold_labels):
        raise RuntimeError("The system output and gold data differ in size: {} vs {}.".format(
            len(system), len(gold_labels)))

    iou = 0
    for i in range(len(gold_labels)):
        system_label, *system_mask = map(int, system[i].split())
        if system_label == gold_labels[i]:
            system_mask = np.array(system_mask, dtype=gold_masks[i].dtype).reshape(gold_masks[i].shape)
            system_pixels = np.sum(system_mask)
            gold_pixels = np.sum(gold_masks[i])
            intersection_pixels = np.sum(system_mask * gold_masks[i])
            iou += intersection_pixels / (system_pixels + gold_pixels - intersection_pixels)

    accuracy = 100 * iou / len(gold_labels)

    print("{}: {:.2f}%".format(system_path, accuracy))
