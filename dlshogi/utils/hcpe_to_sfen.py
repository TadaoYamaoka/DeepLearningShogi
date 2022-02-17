from cshogi import *
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('hcpe')
parser.add_argument('sfen')
args = parser.parse_args()

hcpes = np.fromfile(args.hcpe, dtype=HuffmanCodedPosAndEval)

board = Board()
with open(args.sfen, 'w', newline='\n') as f:
    for hcpe in hcpes:
        board.set_hcp(hcpe['hcp'])
        f.write(board.sfen() + '\n')
