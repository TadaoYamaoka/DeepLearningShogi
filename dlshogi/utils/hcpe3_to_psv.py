from cshogi import *
from dlshogi import cppshogi

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
parser.add_argument('psv')
parser.add_argument('--use_evalfix', action='store_true')
parser.add_argument('--eval_coef', type=float, default=600.0)
args = parser.parse_args()

def score_to_value(score, a):
    return 1.0 / (1.0 + np.exp(-score / a))

def load_hcpe3(path, use_evalfix=False, eval_coef=600.0):
    if use_evalfix:
        from scipy.optimize import curve_fit

    if use_evalfix:
        eval, result = cppshogi.hcpe3_prepare_evalfix(path)
        if (eval == 0).all():
            a = 0
            logging.info('{}, skip evalfix'.format(path))
        else:
            popt, _ = curve_fit(score_to_value, eval, result, p0=[300.0])
            a = popt[0]
            print(f'a={a}')
    else:
        a = eval_coef
    # cppshogi.load_hcpe3の内部で756.0864962951762を基準にしているため調整
    a *= 756.0864962951762 / eval_coef
    sum_len, len_ = cppshogi.load_hcpe3(path, False, a, 1.0)
    if len_ == 0:
        raise RuntimeError('read error')
    return sum_len, len_

positions, _ = load_hcpe3(args.hcpe3, args.use_evalfix, args.eval_coef)

board = Board()
out = open(args.psv, 'wb')
hcpe = np.zeros(1, HuffmanCodedPosAndEval)
psv = np.zeros(1, PackedSfenValue)
for i in range(positions):
    cppshogi.hcpe3_get_hcpe(i, hcpe)
    board.set_hcp(hcpe['hcp'])
    assert board.is_ok()
    board.to_psfen(psv['sfen'])
    psv['score'] = hcpe['eval']
    psv['move'] = move16_to_psv(hcpe[0]['bestMove16'])
    gameResult = hcpe[0]['gameResult']
    # gameResult -> 0: DRAW, 1: BLACK_WIN, 2: WHITE_WIN
    if board.turn == gameResult - 1:
        psv['game_result'] = 1
    elif board.turn == 2 - gameResult:
        psv['game_result'] = -1
    else:
        psv['game_result'] = 0

    psv.tofile(out)

print('positions', positions)
