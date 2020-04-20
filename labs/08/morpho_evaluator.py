#!/usr/bin/env python3
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("gold", type=str, help="Path to gold data.")
parser.add_argument("system", type=str, help="Path to system output.")
args = parser.parse_args()

with open(args.system, "r", encoding="utf-8") as system_file:
    system = [line.rstrip("\n") for line in system_file]

with open(args.gold, "r", encoding="utf-8") as gold_file:
    gold = [line.rstrip("\n") for line in gold_file]

if len(system) < len(gold):
    raise RuntimeError("The system output is shorter than gold data: {} vs {}.".format(len(system), len(gold)))

words, correct_lemmas, correct_tags = 0, 0, 0
for i in range(len(gold)):
    if not gold[i]: continue

    _, gold_lemma, gold_tag = gold[i].split("\t")
    _, system_lemma, system_tag = system[i].split("\t")

    words += 1
    correct_lemmas += gold_lemma == system_lemma
    correct_tags += gold_tag == system_tag

print("Lemma accuracy: {:.2f}%".format(100 * correct_lemmas / words))
print("Tag accuracy: {:.2f}%".format(100 * correct_tags / words))
