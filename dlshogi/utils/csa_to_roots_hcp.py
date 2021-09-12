import numpy as np
from cshogi import *
import pandas as pd
import glob
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('hcp')
parser.add_argument('--moves', type=int, default=16)
parser.add_argument('--filter_moves', type=int, default=80)
parser.add_argument('--filter_rating', type=int, default=3000)
parser.add_argument('--filter_eval', type=int, default=300)
parser.add_argument('--recursive', '-r', action='store_true')
parser.add_argument('--percentile', type=float, default=0.99)

args = parser.parse_args()
filter_rating = args.filter_rating

if args.recursive:
    dir = os.path.join(args.csa_dir, '**')
else:
    dir = args.csa_dir
csa_file_list = glob.glob(os.path.join(dir, '*.csa'), recursive=args.recursive)

board = Board()
parser = Parser()
hcp = np.empty(1, HuffmanCodedPos)
dic = {}
num_games = 0
for filepath in csa_file_list:
    parser.parse_csa_file(filepath)
    if parser.endgame not in ('%TORYO', '%KACHI') or len(parser.moves) < args.filter_moves:
        continue
    if filter_rating > 0 and (parser.ratings[0] < filter_rating and parser.ratings[1] < filter_rating):
        continue
    board.set_sfen(parser.sfen)
    assert board.is_ok(), "{}:{}".format(filepath, parser.sfen)
    for i, (move, score) in enumerate(zip(parser.moves, parser.scores)):
        if i >= args.moves:
            break

        if not board.is_legal(move):
            print("skip {}:{}:{}".format(filepath, i, move_to_usi(move)))
            break

        if board.is_draw():
            break

        if abs(score) > args.filter_eval:
            break

        # hcp
        board.to_hcp(hcp)
        key = hcp.tobytes()
        if not key in dic:
            dic[key] = { 'count': 1, 'ply': board.move_number }
        else:
            dic[key]['count'] += 1

        board.push(move)

    num_games += 1

df = pd.DataFrame.from_dict(dic, orient='index')
print('num_games', num_games)
print(df.describe())

if args.percentile > 0:
    th = int(df['count'].quantile(args.percentile))
    print('th', th)

    df = df[df['count']>=th]
    print('output num', len(df))
    print(df.describe())

hcps = np.empty(len(df), dtype=HuffmanCodedPos)

for i, hcp in enumerate(df.index):
    hcps[i] = np.frombuffer(hcp, HuffmanCodedPos)

hcps.tofile(args.hcp)
