#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("gold_data", type=str, help="Gold data file.")
parser.add_argument("system_data", type=str, help="File produced by a tagger.")
args = parser.parse_args()

gold, system = open(args.gold_data, "r"), open(args.system_data, "r")

correct, total = 0, 0
while True:
    # Read line in both files
    gold_line, system_line = gold.readline(), system.readline()
    if not gold_line and not system_line:
        break
    if (gold_line and not system_line) or (not gold_line and system_line):
        print("The gold data and systems data have different length!")
        exit(1)

    gold_lang, system_lang = gold_line.rstrip("\r\n").split("\t")[0], system_line.rstrip("\r\n").split("\t")[0]

    # Update accuracy
    correct += gold_lang == system_lang
    total += 1

print("Accuracy: {:.2f}".format(100. * correct / total))
