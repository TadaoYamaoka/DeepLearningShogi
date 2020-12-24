from cshogi import *
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('hcpe')
parser.add_argument('psv')
args = parser.parse_args()

hcpes = np.fromfile(args.hcpe, dtype=HuffmanCodedPosAndEval)
psvs = np.zeros(len(hcpes), PackedSfenValue)

board = Board()
for hcpe, psv in zip(hcpes, psvs):
    board.set_hcp(hcpe['hcp'])
    board.to_psfen(psv['sfen'])
    psv['score'] = hcpe['eval']
    psv['move'] = move16_to_psv(hcpe['bestMove16'])
    gameResult = hcpe['gameResult']
    # gameResult -> 0: DRAW, 1: BLACK_WIN, 2: WHITE_WIN
    if board.turn == gameResult - 1:
        psv['game_result'] = 1
    elif board.turn == 2 - gameResult:
        psv['game_result'] = -1

psvs.tofile(args.psv)
