import argparse
import numpy as np
from cshogi import *

parser = argparse.ArgumentParser()
parser.add_argument('psv')
parser.add_argument('hcp')
parser.add_argument('--limit_eval', type=int, default=2000)
parser.add_argument('--limit_moves', type=int)
args = parser.parse_args()

limit_eval = args.limit_eval
limit_moves = args.limit_moves

psvs = np.fromfile(args.psv, dtype=PackedSfenValue)

print(f'input num = {len(psvs)}')

hcps = np.zeros(len(psvs), HuffmanCodedPos)
board = Board()

i = 0
for psv in psvs:
    if abs(psv['score']) > limit_eval:
        continue
    if limit_moves is not None and psv['gamePly'] > limit_moves:
        continue
    if board.set_psfen(psv['sfen']) == False:
        print('illegal data:', psv)
        continue
    if board.is_ok() == False:
        print('illegal data', psv)
        continue
    board.to_hcp(np.asarray(hcps[i]))
    i += 1

hcps[:i].tofile(args.hcp)

print(f'output num = {i}')
