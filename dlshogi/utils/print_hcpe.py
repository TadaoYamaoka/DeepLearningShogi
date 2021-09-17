from cshogi import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('hcpe')
args = parser.parse_args()

hcpes = np.fromfile(args.hcpe, HuffmanCodedPosAndEval)
board = Board()

for hcpe in hcpes:
    board.set_hcp(hcpe['hcp'])
    print(f"{board.sfen()}\t{move_to_usi(hcpe['bestMove16'])}\t{hcpe['eval']}\t{hcpe['gameResult']}")
