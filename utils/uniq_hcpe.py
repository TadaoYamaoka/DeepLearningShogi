import argparse
from cshogi import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('hcpe')
parser.add_argument('hcpe_uniq')
args = parser.parse_args()

hcpes = np.fromfile(args.hcpe, HuffmanCodedPosAndEval)
np.unique(hcpes, axis=0).tofile(args.hcpe_uniq)
