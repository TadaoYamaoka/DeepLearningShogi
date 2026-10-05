import argparse
import os

from dlshogi import cppshogi


def main():
    parser = argparse.ArgumentParser(description="Concatenate hcpe3 caches.")
    parser.add_argument(
        "input_files",
        type=str,
        nargs="+",
        help="input hcpe3 cache files",
    )
    parser.add_argument("--out", "-o", type=str, required=True, help="output hcpe3 cache file")

    args = parser.parse_args()
    if len(args.input_files) < 2:
        parser.error("specify at least two input files")

    out = os.path.abspath(args.out)
    for filepath in args.input_files:
        try:
            same_file = os.path.samefile(filepath, out)
        except FileNotFoundError:
            same_file = os.path.abspath(filepath) == out
        if same_file:
            parser.error("--out must be different from every input file")

    cppshogi.hcpe3_concat_cache(args.input_files, out)


if __name__ == "__main__":
    main()
