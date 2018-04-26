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

total, same_lemma, same_tag = 0, 0, 0
for i in range(len(gold)):
    if not gold[i]: continue
    _, gold_lemma, gold_tag = gold[i].split("\t", 2)

    total += 1

    system_columns = system[i].split("\t", 2)
    if len(system_columns) != 3: continue
    _, system_lemma, system_tag = system_columns

    same_lemma += gold_lemma == system_lemma
    same_tag += gold_tag == system_tag

print("Lemma accuracy: {:.2f}%".format(100 * same_lemma / total))
print("Tag accuracy: {:.2f}%".format(100 * same_tag / total))
