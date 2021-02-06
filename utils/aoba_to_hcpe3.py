from cshogi import *
from cshogi import CSA
import numpy as np
import os
import glob
import lzma
import gzip
import math
import argparse

HuffmanCodedPosAndEval3 = np.dtype([
    ('hcp', dtypeHcp),
    ('eval', dtypeEval),
    ('bestMove16', dtypeMove16),
    ('result', np.uint8),
    ('seq', np.uint8),
    ('candidateNum', np.uint16),
    ])
MoveVisits = np.dtype([
    ('move16', dtypeMove16),
    ('visits', np.uint16),
    ])

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('out_dir')
args = parser.parse_args()

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa*'), recursive=True)
os.makedirs(args.out_dir, exist_ok=True)

hcpes = np.empty(512, HuffmanCodedPosAndEval3)

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

    f = gzip.open(os.path.join(args.out_dir, os.path.splitext(os.path.basename(filepath))[0] + '.hcpe3.gz'), 'wb', compresslevel=6)

    for kif in CSA.Parser.parse_file(file):
        if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI', '%HIKIWAKE', '%CHUDAN') or len(kif.moves) <= 30:
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

        move_visits_list = []

        # 最後に最善手以外が指された局面から開始する
        board.set_sfen(kif.sfen)
        endgame = kif.endgame
        p = 0
        for i, (move, comment) in enumerate(zip(kif.moves, kif.comments)):
            if i <= start:
                board.push(move)
                continue
            # 5手詰みチェック
            if board.mate_move(5) != 0:
                if kif.win != board.turn + 1:
                    # 詰みを見逃して逆転したゲームの結果を修正
                    hcpes[:p]['result'] = board.turn + 1
                    endgame = '%TORYO'
                break
            comments = comment.decode('ascii').split(',')
            candidates = comments[1:] if comments[0].startswith('v=') else comments
            hcpe = hcpes[p]
            board.to_hcp(hcpe['hcp'])
            hcpe['bestMove16'] = move16(move)
            hcpe['result'] = kif.win
            hcpe['seq'] = p // 2
            assert(len(candidates) % 2 == 1)
            candidate_num = (len(candidates) - 1) // 2
            hcpe['candidateNum'] = candidate_num
            move_visits = np.empty(candidate_num, MoveVisits)
            for j, (csa, visits) in enumerate(zip(candidates[1::2], candidates[2::2])):
                m = board.move_from_csa(csa)
                assert(board.is_legal(m))
                move_visits[j]['move16'] = move16(m)
                move_visits[j]['visits'] = visits
            move_visits_list.append(move_visits)
            p += 1
            board.push(move)

        if endgame == '%SENNICHITE':
            hcpes[:p]['result'] += 4
        elif endgame == '%KACHI':
            hcpes[:p]['result'] += 8
        elif endgame == '%HIKIWAKE' or endgame == '%CHUDAN':
            continue

        assert(p <= 512)
        for i in range(p):
            f.write(hcpes[i].tobytes())
            f.write(move_visits_list[i].tobytes())
        position_num += p
    f.close()

print('kif_num', kif_num)
print('position_num', position_num)
