import argparse
import logging
import sys

from dlshogi.data_loader import Hcpe3DataLoader


def main():
    parser = argparse.ArgumentParser(description="Make hcpe3 cache.")
    parser.add_argument("files", type=str, nargs="+", help="hcpe3 files")
    parser.add_argument("--cache", type=str, required=True, help="cache file")
    parser.add_argument("--use_average", action="store_true")
    parser.add_argument("--use_evalfix", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--patch", type=str, help="Overwrite with the hcpe")
    parser.add_argument("--log", help="log file path")

    args = parser.parse_args()

    if args.log:
        logging.basicConfig(
            format="%(asctime)s\t%(levelname)s\t%(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            filename=args.log,
            level=logging.DEBUG,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s\t%(levelname)s\t%(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            stream=sys.stdout,
            level=logging.DEBUG,
        )

    data_len, actual_len = Hcpe3DataLoader.load_files(
        args.files,
        args.use_average,
        args.use_evalfix,
        args.temperature,
        args.patch,
        args.cache,
    )
    if args.use_average:
        logging.info("position num before preprocessing = {}".format(actual_len))
    logging.info("position num = {}".format(data_len))


if __name__ == "__main__":
    main()
