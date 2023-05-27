from cshogi import *
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('yanebook')
parser.add_argument('book')
args = parser.parse_args()

ptn = re.compile(r'^([0-9PLNSGRB][a-i*][0-9][a-i]\+?) ([0-9PLNSGRB][a-i*][0-9][a-i]\+?|none) (-?\d+) (\d+) (\d+)')

board = Board()

count = 0
dic = {}

for line in open(args.yanebook, encoding='utf_8_sig'):
    if line[0] == '#':
        continue
    if line[:4] == "sfen":
        board.set_sfen(line[5:].strip())
        key = board.book_key()
        entries = []
        dic[key] = entries
    else:
        m = ptn.match(line)
        if m:
            usi_move = m.group(1)
            move = board.move_from_usi(usi_move)
            if board.is_legal(move):
                # move, count, score
                entries.append((move, m.group(5), m.group(3)))
                count += 1
        else:
            raise

print(count)

book = np.empty(count, BookEntry)
i = 0
for key in sorted(dic.keys()):
    for entry in dic[key]:
        book_entry = book[i]
        book_entry['key'] = key
        book_entry['fromToPro'] = move16(entry[0])
        book_entry['count'] = entry[1]
        book_entry['score'] = entry[2]
        i += 1

book[:i].tofile(args.book)
