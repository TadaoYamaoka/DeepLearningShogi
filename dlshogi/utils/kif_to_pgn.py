import argparse
import os
import glob

from cshogi import *
from cshogi import KIF
from cshogi import PGN

parser = argparse.ArgumentParser()
parser.add_argument('kif_dir', type=str)
parser.add_argument('pgn', type=str)
args = parser.parse_args()

pgn = PGN.Exporter(args.pgn)

for filepath in glob.glob(os.path.join(args.kif_dir, '**', '*.kif'), recursive=True):
    kif = KIF.Parser.parse_file(filepath)[0]
    win = kif['win']
    if win:
        pgn.tag_pair(
            kif['names'],
            BLACK_WIN if kif['win'] == 'b' else (WHITE_WIN if kif['win'] == 'w' else DRAW),
            starttime=kif['starttime'])

pgn.close()
