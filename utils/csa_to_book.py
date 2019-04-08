from cshogi import *
import argparse
import os
import sys
import glob
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('book')
parser.add_argument('--limit_turn', type=int, default=80)
parser.add_argument('--lower_limit_entries', type=int, default=30)
parser.add_argument('--lower_limit_count', type=int, default=10)
args = parser.parse_args()

csa_file_list = glob.glob(os.path.join(args.dir, '**', '*.csa'), recursive=True)

board = Board()
parser = Parser()
bookdic = {}
for filepath in csa_file_list:
    parser.parse_csa_file(filepath.encode('utf-8'))
    board.set_sfen(parser.sfen)
    assert board.is_ok(), "{}:{}".format(filepath, parser.sfen)
    skip = False
    for i, move in enumerate(parser.moves):
        if i > args.limit_turn:
            break

        if not board.is_legal(move):
            print("skip {}:{}:{}".format(filepath, i, move_to_usi(move)))
            skip = True
            break

        key = board.book_key()
        if key in bookdic:
            bookdic[key][move16(move)] += 1
        else:
            bookdic[key] = defaultdict(int)

        board.push(move)

    if skip:
        continue

# 閾値以下のエントリを削除
entry_num = 0
for key in list(bookdic.keys()):
    entries = bookdic[key]
    sum_count = 0
    entries_keys = list(entries.keys())
    for move in entries_keys:
        sum_count += entries[move]

    if sum_count <= args.lower_limit_entries:
        del bookdic[key]
        continue

    if sum_count > args.lower_limit_count * 100:
        for move in entries_keys:
            if entries[move] <= args.lower_limit_count:
                del entries[move]
            else:
                entry_num += 1
    else:
        entry_num += len(entries)

print('entry_num : {}'.format(entry_num))

# 保存
book_entries = np.empty(entry_num, dtype=BookEntry)
i = 0
for key in sorted(bookdic.keys()):
    entries = bookdic[key]
    for move in entries.keys():
        book_entries[i] = key, move, entries[move], 0
        i += 1
book_entries.tofile(args.book)
