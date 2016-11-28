#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("gold_data", type=str, help="Gold data file.")
parser.add_argument("system_data", type=str, help="File produced by a tagger.")
args = parser.parse_args()

gold, system = open(args.gold_data, "r"), open(args.system_data, "r")

correct_lemma, correct_tag, total = 0, 0, 0
while True:
    # Read line in both files
    gold_line, system_line = gold.readline(), system.readline()
    if not gold_line and not system_line:
        break
    if (gold_line and not system_line) or (not gold_line and system_line):
        print("The gold data and systems data have different length!")
        exit(1)

    # Continue if EOS in both files
    gold_line, system_line = gold_line.rstrip("\r\n"), system_line.rstrip("\r\n")
    if not gold_line and not system_line:
        continue

    # Check that word forms do match
    gold_data = gold_line.split("\t")
    system_data = system_line.split("\t")
    if gold_data[0] != system_data[0]:
        print("The gold form '{}' does not match system form '{}'!".format(gold_data[0], system_data[0]))
        exit(1)

    # Update accuracy
    correct_lemma += gold_data[1] == system_data[1]
    correct_tag += gold_data[2] == system_data[2]
    total += 1

print("Lemma accuracy: {:.2f}".format(100. * correct_lemma / total))
print("Tag accuracy: {:.2f}".format(100. * correct_tag / total))
