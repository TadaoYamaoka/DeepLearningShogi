from cshogi import *
from cshogi import CSA
import numpy as np
import os
import glob
import lzma
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('out')
args = parser.parse_args()

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa*'), recursive=True)

hcps = np.zeros(len(csa_file_list) * 10000 * 10, HuffmanCodedPos)

board = Board()
kif_num = 0
position_num = 0
sennichite_num = 0
kachi_num = 0
for filepath in csa_file_list:
    print(filepath)
    p = 0
    if filepath.endswith('.xz'):
        file = lzma.open(filepath, 'rt')
        filepath = filepath[:-3]
    else:
        file = filepath
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

        # 最後に最善手以外が指された局面から開始する
        board.set_sfen(kif.sfen)
        start_p = p
        for i, move in enumerate(kif.moves):
            if i <= start:
                board.push(move)
                continue
            # 開始局面
            if i + 1 == start:
                board.to_hcp(hcps[p:p+1])
                
            # 千日手
            elif kif.endgame == '%SENNICHITE':
                if i > len(kif.moves) - 10:
                    sennichite_num += 1
                    break
                if i > len(kif.moves) - 20:
                    board.to_hcp(hcps[p:p+1])
                    p += 1
            elif kif.endgame == '%KACHI':
                if i > len(kif.moves) - 30:
                    kachi_num += 1
                    break
                if i > len(kif.moves) - 50:
                    board.to_hcp(hcps[p:p+1])
                    p += 1
            else:
                break
            board.push(move)

print('kif_num', kif_num)
print('position_num', p)
print('sennichite_num', sennichite_num)
print('kachi_num', kachi_num)

hcps = np.unique(hcps[:p])
hcps.tofile(args.out)

print('unique position_num', len(hcps))
