import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('hcp')
parser.add_argument('out')
parser.add_argument('--keep-order', action='store_true', help='keep the order of first appearances')
parser.add_argument('--shuffle', action='store_true', help='shuffle after removing duplicates')
args = parser.parse_args()

HuffmanCodedPos = np.dtype([
    ('hcp', np.uint8, 32),
    ])

data = np.fromfile(args.hcp, dtype=HuffmanCodedPos)
print(len(data))

if args.keep_order:
    _, first_indices = np.unique(data, return_index=True)
    data_unique = data[np.sort(first_indices)]
else:
    data_unique = np.unique(data)
print(len(data_unique))

if args.shuffle:
    np.random.shuffle(data_unique)

data_unique.tofile(args.out)
