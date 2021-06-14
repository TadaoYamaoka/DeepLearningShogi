from cshogi import *
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('yanebook')
parser.add_argument('hcp')
args = parser.parse_args()

move_ptn = re.compile('[0-9PLNSGRB][a-i*][0-9][a-i]')

exist = set()

board = Board()

f = open(args.hcp, 'wb')
hcp = np.empty(1, HuffmanCodedPos)
count = 0

for line in open(args.yanebook, encoding='utf_8_sig'):
    if line[0] == '#':
        continue
    if line[:4] == "sfen":
        board.set_sfen(line[5:].strip())
        key = board.zobrist_hash()
        if key in exist:
            continue
        exist.add(key)
        board.to_hcp(hcp)
        hcp.tofile(f)
        count += 1
    else:
        usi_move = line[:4]
        assert move_ptn.match(usi_move), line
        move = board.move_from_usi(usi_move)
        if board.is_legal(move):
            board.push(move)
            if key not in exist:
                exist.add(key)
                board.to_hcp(hcp)
                hcp.tofile(f)
                count += 1
            board.pop()

print(count)
