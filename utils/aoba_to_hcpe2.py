from cshogi import *
from cshogi import CSA
import numpy as np
import os
import glob
import lzma
import math
import pandas as pd
import argparse

HuffmanCodedPosAndEval2 = np.dtype([
    ('hcp', dtypeHcp),
    ('eval', dtypeEval),
    ('bestMove16', dtypeMove16),
    ('result', np.uint8),
    ('seq', np.uint8),
    ])


parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('out_dir')
parser.add_argument('--min_moves', type=int, default=50)
parser.add_argument('--stat')
args = parser.parse_args()

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa*'), recursive=True)
os.makedirs(args.out_dir, exist_ok=True)

hcpes = np.zeros(10000*512, HuffmanCodedPosAndEval2)

board = Board()
kif_num = 0
toryo_num = 0
sennichite_num = 0
nyugyoku_num = 0
chudan_num = 0
minogashi_num = 0
position_num = 0
stats = []
for filepath in csa_file_list:
    print(filepath)
    name = os.path.splitext(os.path.basename(filepath))[0]
    stat = { 'name':name, '%TORYO':0, '%SENNICHITE':0, '%KACHI':0, '%HIKIWAKE':0, '%CHUDAN':0, '%+ILLEGAL_ACTION':0, '%-ILLEGAL_ACTION':0, 'toryo':0, 'sennichite':0, 'nyugyoku':0, 'chudan':0, 'minogashi':0 }
    stat_moves = []
    p = 0
    if filepath.endswith('.xz'):
        file = lzma.open(filepath, 'rt')
        filepath = filepath[:-3]
    else:
        file = filepath
    for kif in CSA.Parser.parse_file(file):
        stat[kif.endgame] += 1
        stat_moves.append(len(kif.moves))
        if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI', '%HIKIWAKE', '%CHUDAN') or len(kif.moves) < args.min_moves:
            continue
        assert len(kif.moves) <= 513, (filepath, kif.endgame, len(kif.moves))
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

        # 最後に最善手以外が指された局面から開始する
        board.set_sfen(kif.sfen)
        endgame = kif.endgame
        start_p = p
        for i, move in enumerate(kif.moves):
            if i <= start:
                board.push(move)
                continue
            # 5手詰みチェック
            if board.mate_move(5) != 0:
                if kif.win != board.turn + 1:
                    # 詰みを見逃して逆転したゲームの結果を修正
                    hcpes[start_p:p]['result'] = board.turn + 1
                    endgame = '%TORYO'
                stat['minogashi'] += 1
                break
            hcpe = hcpes[p]
            board.to_hcp(hcpe['hcp'])
            hcpe['bestMove16'] = move16(move)
            hcpe['result'] = kif.win
            hcpe['seq'] = i // 2
            p += 1
            board.push(move)

        if endgame == '%SENNICHITE':
            hcpes[start_p:p]['result'] += 4
            stat['sennichite'] += 1
        elif endgame == '%KACHI':
            hcpes[start_p:p]['result'] += 8
            stat['nyugyoku'] += 1
        elif endgame == '%HIKIWAKE' or endgame == '%CHUDAN':
            # 引き分けは出力しない
            p = start_p
            stat['chudan'] += 1
        else:
            stat['toryo'] += 1

    hcpes[:p].tofile(os.path.join(args.out_dir, os.path.splitext(os.path.basename(filepath))[0] + '.hcpe2'))
    position_num += p

    toryo_num += stat['toryo']
    sennichite_num += stat['sennichite']
    nyugyoku_num += stat['nyugyoku']
    chudan_num += stat['chudan']
    minogashi_num += stat['minogashi']

    if args.stat:
        stat['kif_num'] = len(stat_moves)
        for v, k in zip(*np.histogram(stat_moves, 52, (0, 520))):
            stat[k] = v
        stats.append(stat)

print('kif_num', kif_num)
print('toryo_num', toryo_num)
print('sennichite_num', sennichite_num)
print('nyugyoku_num', nyugyoku_num)
print('chudan_num', chudan_num)
print('minogashi_num', minogashi_num)
print('position_num', position_num)
if args.stat:
    pd.DataFrame(stats).to_csv(args.stat, index=False)
