from cshogi import *
from cshogi import CSA
import numpy as np
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('hcpe')
parser.add_argument('--out_maxmove', action='store_true')
parser.add_argument('--filter_moves', type=int, default=50)
parser.add_argument('--filter_rating', type=int, default=3500)
args = parser.parse_args()

filter_moves = args.filter_moves
filter_rating = args.filter_rating

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True)

hcpes = np.zeros(513, HuffmanCodedPosAndEval)

f = open(args.hcpe, 'wb')

board = Board()
kif_num = 0
position_num = 0
for filepath in csa_file_list:
    for kif in CSA.Parser.parse_file(filepath):
        if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI', '%JISHOGI') or len(kif.moves) < filter_moves:
            continue
        if filter_rating > 0 and (kif.ratings[0] < filter_rating and kif.ratings[1] < filter_rating):
            continue

        if kif.endgame == '%JISHOGI':
            if not args.out_maxmove:
                continue

        board.set_sfen(kif.sfen)
        try:
            for i, (move, score) in enumerate(zip(kif.moves, kif.scores)):
                assert board.is_legal(move)
                hcpe = hcpes[i]
                board.to_hcp(hcpe['hcp'])
                assert abs(score) <= 100000
                score = min(32767, max(score, -32767))
                hcpe['eval'] = score if board.turn == BLACK else -score
                hcpe['bestMove16'] = move16(move)
                hcpe['gameResult'] = kif.win
                board.push(move)
        except:
            print(f'skip {filepath}:{i}:{move_to_usi(move)}:{score}')
            continue

        hcpes[:i + 1].tofile(f)

        kif_num += 1
        position_num += i + 1

print('kif_num', kif_num)
print('position_num', position_num)
