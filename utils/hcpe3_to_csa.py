from cshogi import *
from cshogi import CSA
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

ENDGAME_SYMBOLS = {
    1 : '%TORYO',
    2 : '%TORYO',
    4 : '%SENNICHITE',
    9 : '%KACHI',
    10: '%KACHI',
    16: '%CHUDAN',
}

parser = argparse.ArgumentParser()
parser.add_argument('hcpe3')
parser.add_argument('csa')
parser.add_argument('--range')
parser.add_argument('--nyugyoku', action='store_true')
parser.add_argument('--aoba', action='store_true')
parser.add_argument('--out_v', action='store_true')
parser.add_argument('--sort_visits', action='store_true')
args = parser.parse_args()

f = open(args.hcpe3, 'rb')

if args.aoba:
    sep = ','
else:
    sep = '\n'

if args.range:
    start_end = args.range.split(':')
    if len(start_end) == 1:
        start = int(start_end[0])
        end = start + 1
    else:
        if start_end[0] == '':
            start = 0
        else:
            start = int(start_end[0])
        if start_end[1] == '':
            end = sys.maxsize
        else:
            end = int(start_end[1])
else:
    start = 0
    end = sys.maxsize

board = Board()
csa = CSA.Exporter(args.csa)
p = 0
while p < end:
    data = f.read(HuffmanCodedPosAndEval3.itemsize)
    if len(data) == 0:
        break
    hcpe = np.frombuffer(data, HuffmanCodedPosAndEval3, 1)[0]
    board.set_hcp(hcpe['hcp'])
    assert board.is_ok()
    move_num = hcpe['moveNum']
    result = hcpe['result']

    need_output = p >= start and (not args.nyugyoku or result & 8 != 0)
    if need_output:
        csa.info(board, comments=[f"moveNum={move_num},result={result},opponent={hcpe['opponent']}"])

    for i in range(move_num):
        move_info = np.frombuffer(f.read(MoveInfo.itemsize), MoveInfo, 1)[0]
        candidate_num = move_info['candidateNum']
        move_visits = np.frombuffer(f.read(MoveVisits.itemsize * candidate_num), MoveVisits, candidate_num)
        move = board.move_from_move16(move_info['selectedMove16'])
        if need_output:
            if candidate_num > 0:
                if args.aoba:
                    if args.out_v:
                        v = 1.0 / (1.0 + math.exp(-move_info['eval'] * 0.0013226))
                        comment = f"v={v:.3f},"
                    else:
                        comment = ''
                    comment += f"{move_visits['visitNum'].sum()}"
                    if args.sort_visits:
                        move_visits = np.sort(move_visits, order='visitNum')[::-1]
                    for move16, visit_num in zip(move_visits['move16'], move_visits['visitNum']):
                        comment += ',' + move_to_csa(board.move_from_move16(move16)) + ',' + str(visit_num)
                else:
                    comment = '** ' + str(move_info['eval'] * (1 - board.turn * 2))
            else:
                if args.aoba or move_info['eval'] == 0:
                    comment = None
                else:
                    comment = '** ' + str(move_info['eval'] * (1 - board.turn * 2))
            csa.move(move, comment=comment, sep=sep)
        board.push(move)
        assert board.is_ok()
    if need_output:
        csa.endgame(ENDGAME_SYMBOLS[hcpe['result']])
    p += 1
