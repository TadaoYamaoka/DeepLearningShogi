import argparse
import os
import glob
from collections import defaultdict
import pandas as pd
from cshogi import *

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('csv', type=str)
parser.add_argument('--rating', type=int, default=3500)
parser.add_argument('--moves', type=int, default=30)
parser.add_argument('--lower_count', type=int, default=20)
args = parser.parse_args()

board = Board()
parser = Parser()

dic = defaultdict(lambda: [0, 0, 0, 0]) # draw, black_win, white_win, moves
MOVES_INDEX = 3

for filepath in glob.glob(os.path.join(args.dir, '**', '*.csa'), recursive=True):
    parser.parse_csa_file(filepath)

    if any([r < args.rating for r in parser.ratings]):
        continue

    if len(parser.moves) < args.moves:
        continue

    if parser.endgame not in ('%TORYO', '%SENNICHITE', '%JISHOGI', '%KACHI'):
        continue

    moves = [move_to_usi(m) for m in parser.moves[:args.moves]]
    win = parser.win
    len_moves = len(parser.moves)
    for i in range(args.moves):
        sfen = ' '.join(moves[:i+1])
        pos = dic[sfen]
        pos[win] += 1
        pos[MOVES_INDEX] += len_moves

df = pd.DataFrame.from_dict(dic, 'index')
df.rename(columns={DRAW:'draw', BLACK_WIN:'black_win', WHITE_WIN:'white_win', MOVES_INDEX:'avr_moves'}, inplace=True)
df['sum'] = df[['draw', 'black_win', 'white_win']].sum(axis=1)
df = df[df['sum'] >= args.lower_count]
df = df.sort_values('sum', ascending=False)
df['avr_moves'] /= df['sum']
df['winrate'] = (df['black_win'] + df['draw'] * 0.5) / df['sum']

df.to_csv(args.csv, columns=('black_win', 'white_win', 'draw', 'sum', 'winrate', 'avr_moves'), index_label='moves')