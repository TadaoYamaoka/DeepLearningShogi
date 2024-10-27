import argparse

from dlshogi import cppshogi


def main():
    parser = argparse.ArgumentParser(description="Merge hcpe3 cache.")
    parser.add_argument("file1", type=str, help="hcpe3 file1")
    parser.add_argument("file2", type=str, help="hcpe3 file2")
    parser.add_argument("out", type=str, help="output hcpe3 file")

    args = parser.parse_args()

    cppshogi.hcpe3_merge_cache(args.file1, args.file2, args.out)


if __name__ == "__main__":
    main()
