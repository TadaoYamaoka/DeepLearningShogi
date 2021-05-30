import argparse
from cshogi import *
import numpy as np
import pandas as pd

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
GAMERESULT_NYUGYOKU = 8

parser = argparse.ArgumentParser()
parser.add_argument('hcpe3')
parser.add_argument('--csv')
args = parser.parse_args()

f = open(args.hcpe3, 'rb')

board = Board()
stats = []
keys = set()
unique_positions = 0
while True:
    data = f.read(HuffmanCodedPosAndEval3.itemsize)
    if len(data) == 0:
        break
    hcpe = np.frombuffer(data, HuffmanCodedPosAndEval3, 1)[0]
    board.set_hcp(hcpe['hcp'])
    assert board.is_ok()
    move_num = hcpe['moveNum']
    result = hcpe['result']
    opponent = hcpe['opponent']

    positions = 0
    sum_candidate_num = 0
    max_candidate_num = 0
    sum_visits = 0
    sum_top_visits = 0
    for i in range(move_num):
        move_info = np.frombuffer(f.read(MoveInfo.itemsize), MoveInfo, 1)[0]
        candidate_num = move_info['candidateNum']
        move_visits = np.frombuffer(f.read(MoveVisits.itemsize * candidate_num), MoveVisits, candidate_num)
        if candidate_num > 0:
            positions += 1
            sum_candidate_num += candidate_num
            if candidate_num > max_candidate_num:
                max_candidate_num = candidate_num
            sum_visits += move_visits['visitNum'].sum()
            sum_top_visits += move_visits['visitNum'].max()
            key = board.zobrist_hash()
            if key not in keys:
                unique_positions += 1
                keys.add(key)
        move = board.move_from_move16(move_info['selectedMove16'])
        board.push(move)
        assert board.is_ok()

    black_win = 1 if (result & 1) == BLACK_WIN else 0
    nyugyoku = 1 if (result & GAMERESULT_NYUGYOKU) != 0 else 0
    stats.append({
        'moves': move_num,
        'black win': black_win,
        'nyugyoku': nyugyoku,
        'opponent': opponent,
        'positions': positions,
        'candidates avr': sum_candidate_num / positions if positions > 0 else 0,
        'max candidates': max_candidate_num,
        'visits avr': sum_visits / positions if positions > 0 else 0,
        'top visits avr': sum_top_visits / positions if positions > 0 else 0,
        })

df = pd.DataFrame(stats)
if args.csv:
    df.to_csv(args.csv)

print(df[['moves', 'black win', 'nyugyoku', 'opponent']].describe())
print(df[['positions', 'candidates avr', 'max candidates', 'visits avr', 'top visits avr']].describe())
sum_poritions = df['positions'].sum()
print('sum positions', sum_poritions)
print('unique positions', unique_positions)
print('unique positions / sum positions', unique_positions / sum_poritions)
