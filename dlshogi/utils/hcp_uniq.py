import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('hcp')
parser.add_argument('out')
args = parser.parse_args()

HuffmanCodedPos = np.dtype([
    ('hcp', np.uint8, 32),
    ])

data = np.fromfile(args.hcp, dtype=HuffmanCodedPos)
print(len(data))

data_unique = np.unique(data)
print(len(data_unique))

data_unique.tofile(args.out)
