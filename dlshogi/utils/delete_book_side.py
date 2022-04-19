import argparse
import numpy as np

from cshogi import *

parser = argparse.ArgumentParser()
parser.add_argument('side', choices=['black', 'white'], help='side to delete')
parser.add_argument('book')
parser.add_argument('out')
parser.add_argument('--opponent')
args = parser.parse_args()

book = np.fromfile(args.book, BookEntry)
side = BLACK if args.side == 'black' else WHITE
if args.opponent:
    opponent = np.fromfile(args.opponent, BookEntry)
else:
    opponent = None

board = Board()

del_keys = []
dup = set()

def walk():
    key = board.book_key()
    if key in dup:
        return
    dup.add(key)
    entries = book[book['key']==key]
    if board.turn == side:
        if len(entries) > 0:
            del_keys.append(key)
            print(board.sfen())
        elif opponent is not None:
            entries = opponent[opponent['key']==key]

    for entry in entries:
        board.push_move16(entry['fromToPro'])
        walk()
        board.pop()

walk()

out = book
for key in del_keys:
    out = out[out['key']!=key]

out.tofile(args.out)
