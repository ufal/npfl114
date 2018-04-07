#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("test_data", type=str, help="Path to test data.")
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

def as_compressed_ascii_encoded(s):
    import base64
    import io
    import lzma

    with io.BytesIO() as lzma_data:
        with lzma.open(lzma_data, mode="wt", encoding="utf-8") as lzma_file:
            lzma_file.write(s)
        return base64.b85encode(lzma_data.getbuffer())

with open("nsketch_transfer_recodex_submission.py", "w", encoding="utf-8") as submission:
    print("# coding=utf-8", file=submission, end="\n\n")

    for i, path in enumerate(args.sources):
        with open(path, "r", encoding="utf-8") as source:
            print("source_{} =".format(i + 1), as_triple_quoted_literal(source.read()), file=submission, end="\n\n")

    with open(args.test_data, "r", encoding="utf-8") as test:
        print("test_data =", as_compressed_ascii_encoded(test.read()), file=submission, end="\n\n")

    print("""if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())""", file=submission)
