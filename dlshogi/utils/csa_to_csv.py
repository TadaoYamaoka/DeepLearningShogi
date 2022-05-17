import argparse
import pandas as pd
import glob
import os

from cshogi import *
from cshogi.CSA import *

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('csv')
parser.add_argument('--filter_rating', type=int)
args = parser.parse_args()

data = []
for path in glob.glob(os.path.join(args.dir, '**', '*.csa'), recursive=True):
    for parser in Parser.parse_file(path):
        if args.filter_rating:
            if parser.ratings[0] < args.filter_rating or parser.ratings[1] < args.filter_rating:
                continue

        data.append({
            'path': path,
            'name1': parser.names[0],
            'name2': parser.names[1],
            'rating1': parser.ratings[0],
            'rating2': parser.ratings[1],
            'moves': ' '.join([move_to_usi(move) for move in parser.moves]),
            'len': len(parser.moves),
            'endgame': parser.endgame,
            'win': parser.win
        })

pd.DataFrame(data).to_csv(args.csv, index=False)
