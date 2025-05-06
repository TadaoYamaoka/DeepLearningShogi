import argparse
from cshogi import Board, BookEntry, move_to_usi
import numpy as np

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command", required=True)

parser_print = subparsers.add_parser("print")
parser_print.add_argument("book")
parser_print.add_argument("sfen")

parser_order = subparsers.add_parser("order")
parser_order.add_argument("book")
parser_order.add_argument("sfen")
parser_order.add_argument("index", type=int)
parser_order.add_argument("out")

args = parser.parse_args()


def print_book(path, sfen):
    book = np.fromfile(path, BookEntry)
    board = Board(sfen=sfen)
    key = board.book_key()
    print(f"key = {key}")
    print("index\tmove\tcount\tscore")
    for i, row in enumerate(book[book["key"] == key]):
        print(f"{i}\t{move_to_usi(row[1])}\t{row[2]}\t{row[3]}")


def order_book(path, sfen, index, out_path):
    book = np.fromfile(path, BookEntry)
    board = Board(sfen=sfen)
    key = board.book_key()
    entries = book[book["key"] == key]
    count0 = entries[0]["count"]
    entry_target = entries[index].copy()
    entry_target["count"] = count0
    for i in range(index, 0, -1):
        entry = entries[i - 1].copy()
        if entry["count"] == count0:
            entry["count"] -= 1
        entries[i] = entry
    entries[0] = entry_target
    book[book["key"] == key] = entries
    book.tofile(out_path)

    print_book(out_path, sfen)


if args.command == "print":
    print_book(args.book, args.sfen)
elif args.command == "order":
    order_book(args.book, args.sfen, args.index, args.out)
