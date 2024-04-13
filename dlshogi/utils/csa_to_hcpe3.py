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
parser.add_argument('--out_noeval', action='store_true')
parser.add_argument('--out_mate', action='store_true')
parser.add_argument('--out_brinkmate', action='store_true')
parser.add_argument('--uniq', action='store_true')
parser.add_argument('--filter_moves', type=int, default=50)
parser.add_argument('--filter_rating', type=int, default=3800)
parser.add_argument('--filter_win_name') # 指定した対局者が勝った棋譜
parser.add_argument('--skip_opening', action='store_true')
args = parser.parse_args()

filter_moves = args.filter_moves
filter_rating = args.filter_rating
filter_win_name = args.filter_win_name

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True)

hcpe = np.zeros(1, HuffmanCodedPosAndEval3)
move_info_vec = np.empty(513, MoveInfo)
move_visits_vec = np.empty(513, MoveVisits)
move_visits_vec['visitNum'] = 1

f = open(args.hcpe3, 'wb')

board = Board()
kif_num = 0
position_num = 0
duplicates = set()
for filepath in csa_file_list:
    for kif in CSA.Parser.parse_file(filepath):
        endgame = kif.endgame
        if endgame not in ('%TORYO', '%SENNICHITE', '%KACHI', '%JISHOGI') or len(kif.moves) < filter_moves:
            continue
        if filter_rating > 0 and min(kif.ratings) < filter_rating:
            continue
        if filter_win_name and (kif.win == DRAW or kif.names[kif.win - 1] != filter_win_name):
            continue
        # 評価値がない棋譜を除外
        if all(comment == '' for comment in kif.comments[0::2]) or all(comment == '' for comment in kif.comments[1::2]):
            continue
        # 重複削除
        if args.uniq:
            dup_key = ''.join([move_to_usi(move) for move in kif.moves])
            if dup_key in duplicates:
                print(f'duplicate {filepath}')
                continue
            duplicates.add(dup_key)

        hcpe['result'] = kif.win
        if endgame == '%SENNICHITE':
            hcpe['result'] += 4
        elif endgame == '%KACHI':
            hcpe['result'] += 8
        elif endgame == '%JISHOGI':
            if not args.out_maxmove:
                continue
            hcpe['result'] += 16
            
        try:
            if args.out_brinkmate:
                brinkmate_i = -1
                if endgame == '%TORYO':
                    board.set_sfen(kif.sfen)
                    for move in kif.moves:
                        assert board.is_legal(move)
                        board.push(move)
                    while board.is_check():
                        board.pop()
                        board.pop()
                    brinkmate_i = board.move_number

            move_info_vec['candidateNum'] = 1

            board.set_sfen(kif.sfen)
            board.to_hcp(hcpe['hcp'])
            for i, (move, score, comment) in enumerate(zip(kif.moves, kif.scores, kif.comments)):
                assert board.is_legal(move)
                move_info = move_info_vec[i]
                move_visits = move_visits_vec[i]

                assert abs(score) <= 1000000
                eval = min(32767, max(score, -32767))
                move_info['eval'] = eval if board.turn == BLACK else -eval
                move_info['selectedMove16'] = move16(move)
                if comment == '' and args.skip_opening:
                    move_info['candidateNum'] = 0
                else:
                    move_visits['move16'] = move16(move)
                if args.out_brinkmate:
                    if i == brinkmate_i:
                        break
                elif not args.out_mate and endgame != '%KACHI' and abs(score) >= 100000:
                    break
                board.push(move)
        except:
            print(f'skip {filepath}:{i}:{move_to_usi(move)}:{score}')
            continue

        move_num = i + 1
        hcpe['moveNum'] = move_num

        hcpe.tofile(f)
        for move_info, move_visits in zip(move_info_vec[:move_num], move_visits_vec[:move_num]):
            move_info.tofile(f)
            if move_info['candidateNum'] > 0:
                move_visits.tofile(f)
                position_num += 1
        kif_num += 1

print('kif_num', kif_num)
print('position_num', position_num)
