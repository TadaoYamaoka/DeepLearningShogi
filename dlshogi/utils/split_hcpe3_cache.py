import argparse

from dlshogi import cppshogi


def main():
    parser = argparse.ArgumentParser(description="Split hcpe3 cache.")
    parser.add_argument("cache", type=str, help="input hcpe3 cache file")
    parser.add_argument(
        "--outpath",
        type=str,
        help="output file base path (default: input cache path)",
    )
    parser.add_argument(
        "--split",
        type=int,
        required=True,
        help="number of output cache files",
    )

    args = parser.parse_args()
    if args.split <= 0:
        parser.error("--split must be greater than zero")

    outpath = args.outpath or args.cache
    cppshogi.hcpe3_split_cache(args.cache, outpath, args.split)


if __name__ == "__main__":
    main()
