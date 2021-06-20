import argparse
import os
import glob
from cshogi import *
from cshogi import CSA
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--rating', type=int, default=3800)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

kifu_moves_len = []
sennichite_num = 0
kachi_num = 0
#max_hand_fu = []
for filepath in glob.glob(os.path.join(args.dir, '**', '*.csa'), recursive=True):
    for kifu in CSA.Parser.parse_file(filepath):
        if any([r < args.rating for r in kifu.ratings]):
            continue

        if kifu.endgame not in ('%TORYO', '%SENNICHITE', '%JISHOGI', '%KACHI'):
            continue

        moves_len = len(kifu.moves)
        kifu_moves_len.append(moves_len)
        if kifu.endgame == '%SENNICHITE':
            sennichite_num += 1
        elif kifu.endgame == '%KACHI':
            kachi_num += 1
        #board = Board()
        #max_hand_fu_in_game = 0
        #for move in kifu.moves:
        #    board.push(move)
        #    max_hand_fu_in_game = max(max_hand_fu_in_game, board.pieces_in_hand[BLACK][HPAWN], board.pieces_in_hand[WHITE][HPAWN])
        #max_hand_fu.append(max_hand_fu_in_game)

df = pd.DataFrame(kifu_moves_len, columns=['moves'])
print(df.describe())
print('sennichite : {}'.format(sennichite_num))
print('nyugyoku : {}'.format(kachi_num))
#print('max hand fu : {}'.format(max(max_hand_fu)))

if args.plot:
    df.hist()
    plt.show()

    #plt.hist(max_hand_fu, bins=18, range=(0, 18))
    #plt.show()
