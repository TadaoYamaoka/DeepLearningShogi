import argparse
from cshogi import *
from cshogi import CSA
import numpy as np
import pandas as pd
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('gokakusfen')
parser.add_argument('--moves', type=int, default=24)
parser.add_argument('--eval', type=int, default=170)
parser.add_argument('--eval2', type=int, default=100)
parser.add_argument('--count', type=int, default=5)
parser.add_argument('--percentile', type=float, default=0.9)
parser.add_argument('--filter_moves', type=int, default=50)
parser.add_argument('--filter_rating', type=int, default=3500)
args = parser.parse_args()

filter_moves = args.filter_moves
filter_rating = args.filter_rating

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True)

board = Board()
dic = {}
for filepath in csa_file_list:
    for kif in CSA.Parser.parse_file(filepath):
        endgame = kif.endgame
        if endgame not in ('%TORYO', '%KACHI') or len(kif.moves) < filter_moves:
            continue
        if filter_rating > 0 and min(kif.ratings) < filter_rating:
            continue
        # 評価値がない棋譜を除外
        if all(comment == '' for comment in kif.comments[0::2]) or all(comment == '' for comment in kif.comments[1::2]):
            continue

    board.reset()
    sfen = "startpos moves"
    for i, (move, score) in enumerate(zip(kif.moves, kif.scores)):
        if i == args.moves:
            if not sfen in dic:
                dic[sfen] = [abs(score)]
            else:
                dic[sfen].append(abs(score))
            break

        if not board.is_legal(move):
            print("skip {}:{}:{}".format(filepath, i, move_to_usi(move)))
            break

        if abs(score) > args.eval:
            break

        board.push(move)
        sfen += " " + move_to_usi(move)

dic2 = {}
for sfen, eval_list in dic.items():
    dic2[sfen] = { 'count': len(eval_list), 'eval' : np.percentile(eval_list, args.percentile) }

df = pd.DataFrame.from_dict(dic2, orient='index')
print('all num', len(df))
print(df.describe())

df = df[(df['count']>=args.count)&(df['eval']<=args.eval2)]
print('output num', len(df))
print(df.describe())

df.index.to_series().to_csv(args.gokakusfen, header=False, index=False)
