import argparse
from cshogi import *
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('hcpe')
parser.add_argument('split', type=int)
args = parser.parse_args()

hcpes = np.fromfile(args.hcpe, HuffmanCodedPosAndEval)
print(args.hcpe, len(hcpes))

basepath, ext = os.path.splitext(args.hcpe)
num = len(hcpes) // args.split
pos = 0
for i in range(args.split):
    pos_next = pos + num
    if i == args.split - 1:
        pos_next += 1
    hcpes_splited = hcpes[pos:pos_next]
    hcpes_uniq = np.unique(hcpes_splited, axis=0)
    filepath = basepath + f'-{i:03}' + ext
    np.unique(hcpes_uniq, axis=0).tofile(filepath)
    print(filepath, len(hcpes_splited), len(hcpes_uniq))
    pos = pos_next
