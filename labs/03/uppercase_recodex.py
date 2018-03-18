#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("test_data", type=str, help="Path to uppercased test data.")
parser.add_argument("sources", type=str, nargs="+", help="Path to sources.")
args = parser.parse_args()

def as_triple_quoted_literal(s):
    res = ['"'] * 3
    for char in s:
        if char == '"':
            res.append('\\"')
        elif char == "\\":
            res.append("\\\\")
        else:
            res.append(char)
    res += ['"'] * 3

    return "".join(res)

with open("uppercase_recodex_submission.py", "w", encoding="utf-8") as submission:
    for i, path in enumerate(args.sources):
        with open(path, "r", encoding="utf-8") as source:
            print("source_{} =".format(i + 1), as_triple_quoted_literal(source.read()), file=submission, end="\n\n")

    with open(args.test_data, "r", encoding="utf-8") as test:
        print("test_data =", as_triple_quoted_literal(test.read()), file=submission, end="\n\n")

    print("print(test_data, end=\"\")", file=submission)
