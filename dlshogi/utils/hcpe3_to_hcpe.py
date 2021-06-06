from cshogi import *
import numpy as np
import sys
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
parser.add_argument('hcpe3')
parser.add_argument('hcpe')
parser.add_argument('--uniq', action='store_true')
args = parser.parse_args()

f = open(args.hcpe3, 'rb')

board = Board()
out = open(args.hcpe, 'wb')
hcpes = np.zeros(513, HuffmanCodedPosAndEval)
games = 0
positions = 0
while True:
    data = f.read(HuffmanCodedPosAndEval3.itemsize)
    if len(data) == 0:
        break
    hcpe3 = np.frombuffer(data, HuffmanCodedPosAndEval3, 1)[0]
    board.set_hcp(hcpe3['hcp'])
    assert board.is_ok()
    move_num = hcpe3['moveNum']
    result = hcpe3['result'] & 3

    p = 0
    for i in range(move_num):
        move_info = np.frombuffer(f.read(MoveInfo.itemsize), MoveInfo, 1)[0]
        candidate_num = move_info['candidateNum']
        f.seek(MoveVisits.itemsize * candidate_num, 1)
        move = board.move_from_move16(move_info['selectedMove16'])
        if candidate_num > 0:
            hcpe = hcpes[p]
            board.to_hcp(hcpe['hcp'])
            hcpe['eval'] = move_info['eval']
            hcpe['bestMove16'] = move_info['selectedMove16']
            hcpe['gameResult'] = result
            p += 1
        board.push(move)
        assert board.is_ok()

    hcpes[:p].tofile(out)
    games += 1
    positions += p

print('games', games)
print('positions', positions)

if args.uniq:
    hcpes = np.fromfile(args.hcpe, HuffmanCodedPosAndEval)
    hcpes_uniq = np.unique(hcpes, axis=0)
    print('unique positions', len(hcpes_uniq))
    hcpes_uniq.tofile(args.hcpe)
