from cshogi import *
from cshogi import CSA
import numpy as np
import os
import glob
import math
import random
import argparse

HuffmanCodedPosAndEval2 = np.dtype([
    ('hcp', dtypeHcp),
    ('eval', dtypeEval),
    ('bestMove16', dtypeMove16),
    ('result', np.uint8),
    ('dummy', np.uint8),
    ])


parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('out_train')
parser.add_argument('out_test')
parser.add_argument('--filter_moves', type=int, default=80)
parser.add_argument('--filter_rating', type=int, default=3000)
parser.add_argument('--start_moves', type=int, default=30)
parser.add_argument('--test_ratio', type=float, default=0.1)
args = parser.parse_args()

start_moves = args.start_moves

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True)

random.shuffle(csa_file_list)
train_num = int(len(csa_file_list) * (1 - args.test_ratio))
csa_file_list_train = csa_file_list[:train_num]
csa_file_list_test = csa_file_list[train_num:]

def make_hcpe2(csa_file_list, out):
    hcpes = np.zeros(len(csa_file_list)*256, HuffmanCodedPosAndEval2)

    board = Board()
    kif_num = 0
    p = 0
    for filepath in csa_file_list:
        kif = CSA.Parser.parse_file(filepath)[0]
        endgame = kif.endgame
        if endgame not in ('%TORYO', '%SENNICHITE', '%KACHI') or len(kif.moves) < args.filter_moves:
            continue
        if args.filter_rating > 0 and (kif.ratings[0] < args.filter_rating and kif.ratings[1] < args.filter_rating):
            continue
        kif_num += 1

        board.set_sfen(kif.sfen)
        start_p = p
        for i, move in enumerate(kif.moves):
            if not board.is_legal(move):
                print("skip {}:{}:{}".format(filepath, i, move_to_usi(move)))
                break

            if i <= start_moves:
                board.push(move)
                continue

            if board.is_draw() and endgame == '%SENNICHITE':
                break

            hcpe = hcpes[p]
            board.to_hcp(hcpe['hcp'])
            hcpe['bestMove16'] = move16(move)
            hcpe['result'] = kif.win
            p += 1
            board.push(move)

        if endgame == '%SENNICHITE':
            hcpes[start_p:p]['result'] += 4
        elif endgame == '%KACHI':
            hcpes[start_p:p]['result'] += 8

    hcpes[:p].tofile(out)

    print('kif_num', kif_num)
    print('position_num', p)

make_hcpe2(csa_file_list_train, args.out_train)
make_hcpe2(csa_file_list_test, args.out_test)