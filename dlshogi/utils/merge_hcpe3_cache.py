import argparse
import os
import tempfile

from dlshogi import cppshogi


def main():
    parser = argparse.ArgumentParser(description="Merge hcpe3 cache.")
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

    input_files = args.input_files
    out = os.path.abspath(args.out)
    out_dir = os.path.dirname(out)
    for filepath in input_files:
        try:
            same_file = os.path.samefile(filepath, out)
        except FileNotFoundError:
            same_file = os.path.abspath(filepath) == out
        if same_file:
            parser.error("--out must be different from every input file")

    # 2ファイルずつ順番にマージし、最後だけ指定された出力先へ直接書き込む。
    with tempfile.TemporaryDirectory(prefix=".merge_hcpe3_cache-", dir=out_dir) as temp_dir:
        merged = input_files[0]
        for i, filepath in enumerate(input_files[1:]):
            is_last = i == len(input_files) - 2
            next_merged = out if is_last else os.path.join(temp_dir, f"merged-{i:03}.cache")
            cppshogi.hcpe3_merge_cache(merged, filepath, next_merged)
            if i > 0:
                os.remove(merged)
            merged = next_merged


if __name__ == "__main__":
    main()
