#!/usr/bin/env python
import argparse
from deid_pipeline.pipeline import deid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.input, "r") as fin:
        text = fin.read()
    with open(args.output, "w") as fout:
        fout.write(deid(text))
    print("Done:", args.output)

if __name__ == "__main__":
    main()
