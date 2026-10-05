import argparse
from cshogi import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('sfen')
parser.add_argument('hcp')
parser.add_argument('--max', type=int, default=1000000)
parser.add_argument('--uniq', action='store_true', help='remove duplicate positions')
parser.add_argument('--keep-order', action='store_true', help='keep the order of first appearances when using --uniq')
parser.add_argument('--shuffle', action='store_true', help='shuffle positions before writing')

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

hcps = hcps[:p]
if args.uniq:
    if args.keep_order:
        _, first_indices = np.unique(hcps, axis=0, return_index=True)
        hcps = hcps[np.sort(first_indices)]
    else:
        hcps = np.unique(hcps, axis=0)

if args.shuffle:
    np.random.shuffle(hcps)

hcps.tofile(args.hcp)
