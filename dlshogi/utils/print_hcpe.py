from cshogi import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('hcpe')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int)
args = parser.parse_args()

hcpes = np.fromfile(args.hcpe, HuffmanCodedPosAndEval)
board = Board()

start = args.start if args.start >= 0 else len(hcpes) + args.start
end = args.end if args.end is not None else len(hcpes)
end = end if end >= 0 else len(hcpes) + end

for hcpe in hcpes[start:end]:
    board.set_hcp(hcpe['hcp'])
    print(f"{board.sfen()}\t{move_to_usi(hcpe['bestMove16'])}\t{hcpe['eval']}\t{hcpe['gameResult']}")
