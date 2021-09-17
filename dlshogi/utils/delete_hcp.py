import argparse
import numpy as np

from cshogi import *

parser = argparse.ArgumentParser(description='Output the unique positions in hcp1 that are not in hcp2.')
parser.add_argument('hcp1')
parser.add_argument('hcp2')
parser.add_argument('out')
args = parser.parse_args()

hcp1 = np.fromfile(args.hcp1, HuffmanCodedPos)
hcp2 = np.fromfile(args.hcp2, HuffmanCodedPos)

hcp3 = np.setdiff1d(hcp1, hcp2)
hcp3.tofile(args.out)

print(args.hcp1, len(hcp1))
print(args.hcp2, len(hcp2))
print(args.out, len(hcp3))
