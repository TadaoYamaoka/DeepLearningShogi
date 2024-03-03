import argparse
import os
import glob
from itertools import zip_longest

from cshogi import *
from cshogi import KIF
from cshogi import CSA

parser = argparse.ArgumentParser()
parser.add_argument("kif_dir")
parser.add_argument("csa_dir")
parser.add_argument("--encoding")
args = parser.parse_args()

for path in glob.glob(os.path.join(args.kif_dir, "**", "*.kif"), recursive=True):
    try:
        kif = KIF.Parser.parse_file(path)
        relpath = os.path.relpath(path, args.kif_dir)
        csa_path = os.path.join(args.csa_dir, relpath)
        dirname, filename = os.path.split(csa_path)
        base, ext = os.path.splitext(filename)
        csa_path = os.path.join(dirname, base + ".csa")
        os.makedirs(dirname, exist_ok=True)
        csa = CSA.Exporter(csa_path, encoding=args.encoding)

        csa.info(init_board=kif.sfen, names=kif.names)

        if len(kif.times) == 0:
            kif.times = [None] * len(kif.moves)
        for move, time in zip(kif.moves, kif.times):
            csa.move(move, time=time)

        csa.endgame(kif.endgame, time=kif.times[-1])
    except Exception as e:
        print(f"skip {path} {e}")
