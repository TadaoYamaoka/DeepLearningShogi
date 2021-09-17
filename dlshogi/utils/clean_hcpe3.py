import argparse
from cshogi import *
import numpy as np

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
parser.add_argument('cleaned')
args = parser.parse_args()

f = open(args.hcpe3, 'rb')
out = open(args.cleaned, 'wb')

board = Board()
games = 0
positions = 0
while True:
    data = f.read(HuffmanCodedPosAndEval3.itemsize)
    if len(data) == 0:
        break
    hcpe = np.frombuffer(data, HuffmanCodedPosAndEval3, 1)[0]
    board.set_hcp(hcpe['hcp'])
    assert board.is_ok()
    move_num = hcpe['moveNum']

    err = False
    move_infos = []
    for i in range(move_num):
        try:
            move_info = np.frombuffer(f.read(MoveInfo.itemsize), MoveInfo, 1)[0]
            candidate_num = move_info['candidateNum']
            if candidate_num > 0:
                positions += 1
            move_visits = np.frombuffer(f.read(MoveVisits.itemsize * candidate_num), MoveVisits, candidate_num)
        except:
            print('read error')
            err = True
            break
        move_infos.append((move_info, move_visits))
        move = board.move_from_move16(move_info['selectedMove16'])
        board.push(move)
        assert board.is_ok()

    if err:
        break

    hcpe.tofile(out)
    for move_info, move_visits in move_infos:
        move_info.tofile(out)
        move_visits.tofile(out)

    games += 1

print(games)
print(positions)
