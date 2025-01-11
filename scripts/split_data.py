import logging
import gzip
import math
import json
import random
import argparse
import re


logger = logging.getLogger("split_data")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", dest="input", help="Input JSONL file")
    parser.add_argument("--first_output", dest="first_output", help="")
    parser.add_argument("--second_output", dest="second_output", help="")
    parser.add_argument("--second_proportion", dest="second_proportion", type=float)
    parser.add_argument("--split_field", dest="split_field")
    parser.add_argument("--random_seed", dest="random_seed", default=None)
    args = parser.parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    data = {}
    count = 0
    with gzip.open(args.input, "rt") as ifd:
        for i, line in enumerate(ifd):
            j = json.loads(line)
            key = j[args.split_field] if args.split_field else i
            data[i] = data.get(i, [])
            data[i].append(j)
            count += 1

    data = list(data.values())
    random.shuffle(data)

    target = int(count * args.second_proportion)
    written = 0
    with gzip.open(args.first_output, "wt") as ofdA, gzip.open(args.second_output, "wt") as ofdB:
        for data_chunk in data:
            if written < target:
                for datum in data_chunk:
                    ofdB.write(json.dumps(datum) + "\n")
                    written += 1
            else:
                for datum in data_chunk:
                    ofdA.write(json.dumps(datum) + "\n")
                    written += 1        
