import argparse
from cshogi import *
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('hcpe', type=str, nargs='+')
parser.add_argument('--outpath')
parser.add_argument('--split', type=int)
parser.add_argument('--positions', type=int)
parser.add_argument('--uniq', action='store_true')
parser.add_argument('--uniq_each_split', action='store_true')
parser.add_argument('--keep-order', action='store_true', help='keep the order of first appearances when removing duplicates')
parser.add_argument('--shuffle-before-split', action='store_true', help='shuffle positions before splitting')
parser.add_argument('--shuffle', action='store_true', help='shuffle each output after removing duplicates')
args = parser.parse_args()

hcpes = np.empty(0, HuffmanCodedPosAndEval)
for hcpe in args.hcpe:
    hcpes = np.concatenate([hcpes, np.fromfile(hcpe, HuffmanCodedPosAndEval)])
num_positions = len(hcpes)

if args.uniq:
    if args.keep_order:
        _, first_indices = np.unique(hcpes, axis=0, return_index=True)
        hcpes = hcpes[np.sort(first_indices)]
    else:
        hcpes = np.unique(hcpes, axis=0)
    print(args.hcpe, num_positions, len(hcpes))
else:
    print(args.hcpe, num_positions)

if args.shuffle_before_split:
    np.random.shuffle(hcpes)

if args.outpath:
    outpath = args.outpath
else:
    outpath = args.hcpe[0]
basepath, ext = os.path.splitext(outpath)

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
        if args.keep_order:
            _, first_indices = np.unique(hcpes_splited, axis=0, return_index=True)
            hcpes_output = hcpes_splited[np.sort(first_indices)]
        else:
            hcpes_output = np.unique(hcpes_splited, axis=0)
    else:
        hcpes_output = hcpes_splited

    if args.shuffle:
        hcpes_output = hcpes_output.copy()
        np.random.shuffle(hcpes_output)

    hcpes_output.tofile(filepath)
    if args.uniq_each_split:
        print(filepath, len(hcpes_splited), len(hcpes_output))
    else:
        print(filepath, len(hcpes_splited))
    pos = pos_next
