#!/usr/bin/env python3
import argparse
import sys

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

parser = argparse.ArgumentParser()
parser.add_argument("system", type=str, help="Path to system output.")
parser.add_argument("gold", type=str, help="Path to gold data.")
args = parser.parse_args()

with open(args.system, "r", encoding="utf-8") as system_file:
    system = [line.rstrip("\n") for line in system_file]

with open(args.gold, "r", encoding="utf-8") as gold_file:
    gold = [line.rstrip("\n") for line in gold_file]

if len(system) < len(gold):
    raise RuntimeError("The system output is shorter than gold data: {} vs {}.".format(len(system), len(gold)))

score = 0
for i in range(len(gold)):
    gold_sentence = gold[i].split(" ")
    system_sentence = system[i].split(" ")
    score += edit_distance(gold_sentence, system_sentence) / len(gold_sentence)

print("Average normalized edit distance: {:.2f}%".format(100 * score / len(gold)))
