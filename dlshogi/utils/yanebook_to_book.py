from cshogi import *
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('yanebook')
parser.add_argument('book')
parser.add_argument('--score_to_count', action='store_true')
parser.add_argument('--temperature', type=float, default=10.0)
parser.add_argument('--flip', action='store_true')
args = parser.parse_args()

ptn = re.compile(r'^([0-9PLNSGRB][a-i*][0-9][a-i]\+?) ([0-9PLNSGRB][a-i*][0-9][a-i]\+?|none|None) (-?\d+) (\d+) (\d+)')

board = Board()
if args.flip:
    board_flip = Board()

count = 0
dic = {}

for line in open(args.yanebook, encoding='utf_8_sig'):
    if line[0] == '#':
        continue
    if line[:4] == "sfen":
        sfen = line[5:].strip()
        board.set_sfen(sfen)
        key = board.book_key()
        entries = []
        dic[key] = entries
        if args.flip:
            sfen_flip = rotate_sfen(sfen)
            board_flip.set_sfen(sfen_flip)
            key_flip = board_flip.book_key()
            entries_flip = []
            dic[key_flip] = entries_flip
    else:
        m = ptn.match(line)
        if m:
            usi_move = m.group(1)
            move = board.move_from_usi(usi_move)
            if board.is_legal(move):
                # move, count, score
                entries.append((move, m.group(5), m.group(3)))
                count += 1
            if args.flip:
                move_flip = move_rotate(move)
                if board_flip.is_legal(move_flip):
                    # move, count, score
                    entries_flip.append((move_flip, m.group(5), m.group(3)))
                    count += 1
        else:
            raise

print(len(dic))
print(count)

book = np.empty(count, BookEntry)
i = 0
if args.score_to_count:
    for key in sorted(dic.keys()):
        entries = dic[key]
        scores = np.array([entry[2] for entry in entries], dtype=np.float32)
        scores -= scores.max()
        exp_scores = np.exp(scores / args.temperature)
        counts = (exp_scores / exp_scores.sum() * 10000).astype(np.uint16)
        for entry, count in zip(entries, counts):
            book_entry = book[i]
            book_entry['key'] = key
            book_entry['fromToPro'] = move16(entry[0])
            book_entry['count'] = count
            book_entry['score'] = entry[2]
            i += 1
else:
    for key in sorted(dic.keys()):
        for entry in dic[key]:
            book_entry = book[i]
            book_entry['key'] = key
            book_entry['fromToPro'] = move16(entry[0])
            book_entry['count'] = entry[1]
            book_entry['score'] = entry[2]
            i += 1

book[:i].tofile(args.book)
