#!/usr/bin/env python3
import argparse
import sys

import numpy as np

from fashion_masks_data import FashionMasks

parser = argparse.ArgumentParser()
parser.add_argument("system", type=str, help="Path to system output.")
parser.add_argument("dataset", type=str, help="Which dataset to evaluate ('dev', 'test').")
args = parser.parse_args()

gold = getattr(FashionMasks(), args.dataset)
gold_labels = gold.data["labels"]
gold_masks = gold.data["masks"]

with open(args.system, "r", encoding="utf-8") as system_file:
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

print("{:.3f}".format(100 * iou / len(gold_labels)))
