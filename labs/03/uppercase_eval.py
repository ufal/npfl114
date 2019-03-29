#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("system", type=str, help="Path to system output.")
parser.add_argument("gold", type=str, help="Path to gold data.")
args = parser.parse_args()

with open(args.system, "r", encoding="utf-8-sig") as system_file:
    system = system_file.read()

with open(args.gold, "r", encoding="utf-8-sig") as gold_file:
    gold = gold_file.read()

same = 0
for i in range(len(gold)):
    if system[i].lower() != gold[i].lower():
        raise RuntimeError("The system output and gold data differ on position {}: '{}' vs '{}'.".format(
            i, system[i:i+20].lower(), gold[i:i+20].lower()))

    same += gold[i] == system[i]

print("{:.2f}".format(100 * same / len(gold)))
