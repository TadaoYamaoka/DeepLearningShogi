import argparse
import numpy as np
import os
import re
from tqdm import tqdm

from cshogi import *
from cshogi.usi import Engine

parser = argparse.ArgumentParser()
parser.add_argument('hcp')
parser.add_argument('start', type=int)
parser.add_argument('end', type=int)
parser.add_argument('delhcp')
parser.add_argument('usi')
parser.add_argument('usi_options')
parser.add_argument('--nodes', type=int, default=80000)
parser.add_argument('--th', type=int, default=2500)
args = parser.parse_args()

with open(args.hcp, 'rb') as f:
    f.seek(args.start * HuffmanCodedPos.itemsize)
    hcps = np.fromfile(f, HuffmanCodedPos, args.end - args.start)

print('read num', len(hcps))
delhcps = np.zeros(len(hcps), HuffmanCodedPos)

os.chdir(os.path.dirname(args.usi))
engine = Engine(args.usi)

for option in args.usi_options.split(','):
    k, v = option.split(':')
    engine.setoption(k, v)

engine.isready(print)

ptn = re.compile(r'score (cp|mate) ([+\-0-9]+)')

class Listener:
    def __init__(self):
        self.info1 = None
        self.info2 = None

    def __call__(self, line):
        self.info1 = self.info2
        self.info2 = line
listener = Listener()

board = Board()
p = 0
for hcp in tqdm(hcps):
    if not board.set_hcp(np.asarray(hcp)):
        delhcps[p] = hcp
        p += 1
        continue
    # 王手がかかっている局面を除く
    if board.is_check():
        delhcps[p] = hcp
        p += 1
        continue
    engine.position(sfen=board.sfen())
    engine.go(nodes=args.nodes, listener=listener)
    m = ptn.search(listener.info1)
    #print(m[1], m[2])
    if m[1] == 'mate':
        eval = 30000
    else:
        eval = abs(int(m[2]))
    if eval > args.th:
        delhcps[p] = hcp
        p += 1

print('delete', p)
delhcps[:p].tofile(args.delhcp)