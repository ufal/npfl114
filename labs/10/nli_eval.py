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

correct = 0
for i in range(len(gold)):
    gold_language = gold[i].split("\t", 1)[0]
    system_language = system[i].split("\t", 1)[0]
    correct += gold_language == system_language

print("Accuracy: {:.2f}%".format(100 * correct / len(gold)))
