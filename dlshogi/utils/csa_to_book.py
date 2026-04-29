from cshogi import *
from cshogi import CSA
import argparse
import os
import sys
import glob
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('book')
parser.add_argument('--limit_moves', type=int, default=80)
parser.add_argument('--limit_entries', type=int, default=50)
parser.add_argument('--filter_rating', type=int)
parser.add_argument('--only_winner', action='store_true')
parser.add_argument('--winner')
parser.add_argument('--yanebook', action='store_true')
args = parser.parse_args()

csa_file_list = glob.glob(os.path.join(args.dir, '**', '*.csa'), recursive=True)

board = Board()
parser = CSA.Parser()
num_games = 0
bookdic = {}
if args.yanebook:
    sfendic = {}
for filepath in csa_file_list:
    try:
        parser.parse_csa_file(filepath)
        if args.filter_rating:
            if parser.win == DRAW:
               if parser.ratings[0] < args.filter_rating or parser.ratings[1] < args.filter_rating:
                continue
            elif parser.ratings[2 - parser.win] < args.filter_rating:
                continue
        if args.only_winner:
            if parser.win == 0:
                continue
            if args.winner and args.winner not in parser.names[parser.win - 1]:
                continue
        board.set_sfen(parser.sfen)
        assert board.is_ok(), "{}:{}".format(filepath, parser.sfen)
        for i, move in enumerate(parser.moves):
            if i >= args.limit_moves:
                break

            if not board.is_legal(move):
                print("skip {}:{}:{}".format(filepath, i, move_to_usi(move)))
                break

            if args.only_winner and board.turn != parser.win - 1:
                board.push(move)
                continue

            key = board.book_key()
            if key not in bookdic:
                bookdic[key] = defaultdict(int)
                if args.yanebook:
                    sfendic[key] = board.sfen()
            bookdic[key][move16(move)] += 1

            board.push(move)

        num_games += 1
    except Exception as e:
        print("skip {} {}".format(filepath, e))

# 閾値以下のエントリを削除
num_positions = 0
num_entries = 0
for key in list(bookdic.keys()):
    entries = bookdic[key]
    sum_count = 0
    for count in entries.values():
        sum_count += count

    if sum_count <= args.limit_entries:
        del bookdic[key]
        continue

    num_positions += 1
    num_entries += len(entries)

print(f"games : {num_games}")
print(f"positions : {num_positions}")
print(f'entries : {num_entries}')

# 保存
if args.yanebook:
    with open(args.book, "w") as f:
        f.write("#YANEURAOU-DB2016 1.00\n")
        for key in sorted(bookdic.keys()):
            f.write(f"sfen {sfendic[key]}\n")
            entries = bookdic[key]
            for move, count in sorted(entries.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{move_to_usi(move)} none 0 0 {count}\n")
else:
    book_entries = np.empty(num_entries, dtype=BookEntry)
    i = 0
    for key in sorted(bookdic.keys()):
        entries = bookdic[key]
        for move, count in entries.items():
            book_entries[i] = key, move, count, 0
            i += 1
    assert i == num_entries
    book_entries.tofile(args.book)
