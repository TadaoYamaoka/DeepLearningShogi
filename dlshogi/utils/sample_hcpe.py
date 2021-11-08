import argparse
from cshogi import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('hcpe')
parser.add_argument('hcpe_sample')
parser.add_argument('num', type=int)
args = parser.parse_args()

hcpes = np.fromfile(args.hcpe, HuffmanCodedPosAndEval)
print(len(hcpes))

np.random.choice(hcpes, args.num, replace=False).tofile(args.hcpe_sample)
print(args.num)
