import argparse
import numpy as np
import os

dtypeHcp = np.dtype((np.uint8, 32))
dtypeEval = np.dtype(np.int16)
dtypeMove16 = np.dtype(np.int16)

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
parser.add_argument('--num_positions', '-n', type=int)
parser.add_argument('--opponent', type=int)
args = parser.parse_args()

if args.num_positions:
    num_positions = args.num_positions
else:
    num_positions = 2147483647 # INT_MAX

basepath, ext = os.path.splitext(args.hcpe3)

f = open(args.hcpe3, 'rb')

n = 0
games = 0
positions = 0
pre_positions = 0
while True:
    data = f.read(HuffmanCodedPosAndEval3.itemsize)
    if len(data) == 0:
        break
    hcpe3 = np.frombuffer(data, HuffmanCodedPosAndEval3, 1)[0]
    move_num = hcpe3['moveNum']
    if args.opponent is not None and hcpe3['opponent'] != args.opponent:
        for i in range(move_num):
            move_info = np.frombuffer(f.read(MoveInfo.itemsize), MoveInfo)[0]
            candidate_num = move_info['candidateNum']
            f.seek(MoveVisits.itemsize * candidate_num, 1)
        continue

    if positions >= n * num_positions:
        if n > 0:
            print(out_path, positions - pre_positions)
        pre_positions = positions
        out_path = f'{basepath}.{n:0>3}{ext}'
        out = open(out_path, 'wb')
        n += 1

    hcpe3.tofile(out)
    for i in range(move_num):
        move_info = np.frombuffer(f.read(MoveInfo.itemsize), MoveInfo, 1)[0]
        move_info.tofile(out)
        candidate_num = move_info['candidateNum']
        assert 0 <= candidate_num <= 593
        if candidate_num > 0:
            candidates = np.frombuffer(f.read(MoveVisits.itemsize * candidate_num), MoveVisits)
            candidates.tofile(out)
            positions += 1

    games += 1

print(out_path, positions - pre_positions)
print('games', games)
print('positions', positions)
