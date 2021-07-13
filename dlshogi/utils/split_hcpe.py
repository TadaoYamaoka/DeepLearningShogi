import argparse
from cshogi import *
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('hcpe')
parser.add_argument('--split', type=int)
parser.add_argument('--positions', type=int)
parser.add_argument('--uniq', action='store_true')
parser.add_argument('--uniq_each_split', action='store_true')
parser.add_argument('--shuffle', action='store_true')
args = parser.parse_args()

hcpes = np.fromfile(args.hcpe, HuffmanCodedPosAndEval)
num_positions = len(hcpes)

if args.uniq:
    hcpes = np.unique(hcpes, axis=0)
    print(args.hcpe, num_positions, len(hcpes))
else:
    print(args.hcpe, num_positions)

if args.shuffle:
    np.random.shuffle(hcpes)

basepath, ext = os.path.splitext(args.hcpe)
if args.split:
    num_split = args.split
    num = len(hcpes) // num_split
elif args.positions:
    num = args.positions
    num_split = (len(hcpes) + num - 1) // num
else:
    num_split = 1
    num = len(hcpes)
pos = 0
for i in range(num_split):
    pos_next = pos + num
    if i == num_split - 1:
        pos_next = len(hcpes)
    hcpes_splited = hcpes[pos:pos_next]
    filepath = basepath + f'-{i+1:03}' + ext
    if args.uniq_each_split:
        hcpes_uniq = np.unique(hcpes_splited, axis=0)
        hcpes_uniq.tofile(filepath)
        print(filepath, len(hcpes_splited), len(hcpes_uniq))
    else:
        hcpes_splited.tofile(filepath)
        print(filepath, len(hcpes_splited))
    pos = pos_next
