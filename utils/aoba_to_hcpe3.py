from cshogi import *
from cshogi import CSA
import numpy as np
import os
import glob
import lzma
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
parser.add_argument('out_dir')
parser.add_argument('--out_maxmove', action='store_true')
args = parser.parse_args()

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa*'), recursive=True)
os.makedirs(args.out_dir, exist_ok=True)

hcpe = np.zeros(1, HuffmanCodedPosAndEval3)
move_infos = np.empty(513, MoveInfo)

board = Board()
kif_num = 0
position_num = 0
for filepath in csa_file_list:
    print(filepath)
    p = 0
    if filepath.endswith('.xz'):
        file = lzma.open(filepath, 'rt')
        filepath = filepath[:-3]
    else:
        file = filepath

    f = open(os.path.join(args.out_dir, os.path.splitext(os.path.basename(filepath))[0] + '.hcpe3'), 'wb')

    for kif in CSA.Parser.parse_file(file):
        if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI', '%CHUDAN') or len(kif.moves) <= 30:
            continue
        kif_num += 1
        board.set_sfen(kif.sfen)
        # 30手までで最善手以外が指された手番を見つける
        start = -1
        for i, (move, comment) in enumerate(zip(kif.moves, kif.comments)):
            comments = comment.decode('ascii').split(',')
            if comments[0].startswith('v='):
                candidates = comments[1:]
            else:
                candidates = comments
            if board.move_from_csa(candidates[1]) != move:
                start = i
            if i >= 29:
                break
            board.push(move)
            assert board.is_ok()

        move_visits_list = []

        # 最後に最善手以外が指された局面から開始する
        board.set_sfen(kif.sfen)
        endgame = kif.endgame
        p = 0
        board.to_hcp(hcpe['hcp'])
        hcpe['result'] = kif.win
        move_num = len(kif.moves)
        for i, (move, comment) in enumerate(zip(kif.moves, kif.comments)):
            move_info = move_infos[i]
            if i <= start:
                move_info['selectedMove16'] = move16(move)
                move_info['eval'] = 0
                move_info['candidateNum'] = 0
                move_visits_list.append(None)
                board.push(move)
                continue
            # 5手詰みチェック
            mate_move = board.mate_move(5)
            if mate_move != 0:
                if kif.win != board.turn + 1:
                    # 詰みを見逃して逆転したゲームの結果を修正
                    hcpe['result'] = board.turn + 1
                endgame = '%TORYO'
                move_info['selectedMove16'] = move16(mate_move)
                move_info['eval'] = 30000
                move_info['candidateNum'] = 0
                move_visits_list.append(None)
                move_num = i + 1
                break
            comments = comment.decode('ascii').split(',')
            if comments[0].startswith('v='):
                candidates = comments[1:]
                v = float(comments[0].split('=')[1])
                if v == 1.0:
                    move_info['eval'] = 30000
                elif v == 0.0:
                    move_info['eval'] = -30000
                else:
                    move_info['eval'] = int(-math.log(1.0 / v - 1.0) * 756.0864962951762)
            else:
                candidates = comments
                move_info['eval'] = 0
            move_info['selectedMove16'] = move16(move)
            assert(len(candidates) % 2 == 1)
            candidate_num = (len(candidates) - 1) // 2
            move_info['candidateNum'] = candidate_num
            move_visits = np.empty(candidate_num, MoveVisits)
            for j, (csa, visit_num) in enumerate(zip(candidates[1::2], candidates[2::2])):
                m = board.move_from_csa(csa)
                assert(board.is_legal(m))
                move_visits[j]['move16'] = move16(m)
                move_visits[j]['visitNum'] = visit_num
            move_visits_list.append(move_visits)
            p += 1
            board.push(move)
            assert board.is_ok()

        if endgame == '%SENNICHITE':
            hcpe['result'] += 4
        elif endgame == '%KACHI':
            hcpe['result'] += 8
        elif endgame == '%CHUDAN':
            if not args.out_maxmove:
                continue
            hcpe['result'] += 16

        assert len(move_visits_list) == move_num
        hcpe['moveNum'] = move_num
        f.write(hcpe.tobytes())
        for i in range(move_num):
            f.write(move_infos[i].tobytes())
            if move_visits_list[i] is not None:
                f.write(move_visits_list[i].tobytes())
        position_num += p
    f.close()

print('kif_num', kif_num)
print('position_num', position_num)
