import argparse
from cshogi import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('sfen')
parser.add_argument('hcp')
parser.add_argument('--max', type=int, default=1000000)

args = parser.parse_args()

board = Board()
hcps = np.empty(args.max, HuffmanCodedPos)
p = 0
for line in open(args.sfen):
    c = line.strip().split(' moves ')
    if c[0] == 'startpos':
        board.reset()
    else:
        board.set_sfen(c[0])

    board.to_hcp(np.asarray(hcps[p]))
    p += 1

    if len(c) >= 2:
        for move in c[1].split(' '):
            board.push_usi(move)
            board.to_hcp(np.asarray(hcps[p]))
            p += 1

np.unique(hcps[:p], axis=0).tofile(args.hcp)
