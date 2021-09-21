import argparse
from cshogi import *
from cshogi import CSA
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('gokakusfen')
parser.add_argument('--moves1', type=int, default=24)
parser.add_argument('--moves2', type=int, default=36)
parser.add_argument('--eval', type=int, default=170)
parser.add_argument('--eval2', type=int, default=100)
parser.add_argument('--filter_moves', type=int, default=50)
parser.add_argument('--filter_rating', type=int, default=3800)
args = parser.parse_args()

filter_moves = args.filter_moves
filter_rating = args.filter_rating

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True)

board = Board()
hcp = np.empty(1, HuffmanCodedPos)
dic = {}
for filepath in csa_file_list:
    for kif in CSA.Parser.parse_file(filepath):
        endgame = kif.endgame
        if endgame not in ('%TORYO', '%SENNICHITE', '%KACHI', '%JISHOGI') or len(kif.moves) < filter_moves:
            continue
        if filter_rating > 0 and min(kif.ratings) < filter_rating:
            continue
        # 評価値がない棋譜を除外
        if all(comment == '' for comment in kif.comments[0::2]) or all(comment == '' for comment in kif.comments[1::2]):
            continue

        board.reset()
        sfen = "startpos moves"
        for i, (move, score) in enumerate(zip(kif.moves, kif.scores)):
            if not board.is_legal(move):
                print("skip {}:{}:{}".format(filepath, i, move_to_usi(move)))
                break

            if i == args.moves1:
                if abs(score) > args.eval2:
                    break
                board.to_hcp(hcp)
                key = hcp.tobytes()
            elif i == args.moves2:
                if  abs(score) > args.eval2:
                    break

                if not key in dic:
                    dic[key] = sfen + '\n'
                break
            elif abs(score) > args.eval:
                break

            board.push(move)
            sfen += " " + move_to_usi(move)

open(args.gokakusfen, 'w').writelines(dic.values())
