from cshogi import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('hcp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int)
args = parser.parse_args()

hcps = np.fromfile(args.hcp, HuffmanCodedPos)
board = Board()

start = args.start if args.start >= 0 else len(hcps) + args.start
end = args.end if args.end is not None else len(hcps)
end = end if end >= 0 else len(hcps) + end

for hcp in hcps[start:end]:
    board.set_hcp(hcp['hcp'])
    print(board.sfen())
