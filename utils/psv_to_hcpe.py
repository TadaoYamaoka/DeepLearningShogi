from cshogi import *
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('psv')
parser.add_argument('hcpe')
args = parser.parse_args()

psvs = np.fromfile(args.psv, dtype=PackedSfenValue)
hcpes = np.zeros(len(psvs), dtype=HuffmanCodedPosAndEval)

print(f'input num = {len(psvs)}')

board = Board()
for psv, hcpe in zip(psvs, hcpes):
    board.set_psfen(psv['sfen'])
    board.to_hcp(hcpe['hcp'])
    hcpe['eval'] = psv['score']
    hcpe['bestMove16'] = move16_from_psv(psv['move'])
    game_result = psv['game_result']
    # gameResult -> 0: DRAW, 1: BLACK_WIN, 2: WHITE_WIN
    if game_result == 1:
        hcpe['gameResult'] = board.turn + 1
    elif game_result == -1:
        hcpe['gameResult'] = 2 - board.turn

hcpes.tofile(args.hcpe)
