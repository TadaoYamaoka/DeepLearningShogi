from cshogi import *
from cshogi import CSA
import numpy as np
import os
import glob
import math
import argparse

HuffmanCodedPosAndEval3 = np.dtype([
    ('hcp', dtypeHcp), # 開始局面
    ('moveNum', np.uint16), # 手数
    ('result', np.uint8), # 結果（xxxxxx11:勝敗、xxxxx1xx:千日手、xxxx1xxx:入玉宣言、xxx1xxxx:最大手数）
    ('opponent', np.uint8), # 対戦相手（0:自己対局、1:先手usi、2:後手usi）
    ])
MoveInfo = np.dtype([
    ('selectedMove16', dtypeMove16), # 指し手
    ('eval', dtypeEval), # 評価値
    ('candidateNum', np.uint16), # 候補手の数
    ])
MoveVisits = np.dtype([
    ('move16', dtypeMove16), # 候補手
    ('visitNum', np.uint16), # 訪問回数
    ])

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('hcpe3')
parser.add_argument('--out_maxmove', action='store_true')
parser.add_argument('--filter_moves', type=int, default=50)
parser.add_argument('--filter_rating', type=int, default=3800)
args = parser.parse_args()

filter_moves = args.filter_moves
filter_rating = args.filter_rating

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True)

hcpe = np.zeros(1, HuffmanCodedPosAndEval3)
move_info = np.empty(1, MoveInfo)
move_info['candidateNum'] = 1
move_visits = np.empty(1, MoveVisits)
move_visits['visitNum'] = 1

f = open(args.hcpe3, 'wb')

board = Board()
kif_num = 0
position_num = 0
for filepath in csa_file_list:
    for kif in CSA.Parser.parse_file(filepath):
        if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI', '%CHUDAN') or len(kif.moves) < filter_moves:
            continue
        if filter_rating > 0 and (kif.ratings[0] < filter_rating and kif.ratings[1] < filter_rating):
            continue

        hcpe['result'] = kif.win
        if kif.endgame == '%SENNICHITE':
            hcpe['result'] += 4
        elif kif.endgame == '%KACHI':
            hcpe['result'] += 8
        elif kif.endgame == '%CHUDAN':
            if not args.out_maxmove:
                continue
            hcpe['result'] += 16

        kif_num += 1
        board.set_sfen(kif.sfen)
        board.to_hcp(hcpe['hcp'])
        move_num = len(kif.moves)
        hcpe['moveNum'] = move_num
        hcpe.tofile(f)
        for i, (move, score) in enumerate(zip(kif.moves, kif.scores)):
            assert board.is_pseudo_legal(move)
            move_info['eval'] = score if board.turn == BLACK else -score
            move_info['selectedMove16'] = move16(move)
            move_info.tofile(f)
            move_visits['move16'] = move16(move)
            move_visits.tofile(f)
            position_num += 1
            board.push(move)

print('kif_num', kif_num)
print('position_num', position_num)
