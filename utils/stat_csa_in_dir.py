﻿import argparse
import os
import sys
import shogi
import shogi.CSA
import statistics
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
args = parser.parse_args()

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

kifu_num = 0
kifu_moves_len = []
max_hand_fu = []
for filepath in find_all_files(args.dir):
    kifu_num += 1
    kifu = shogi.CSA.Parser.parse_file(filepath, encoding='utf-8')[0]
    moves_len = len(kifu['moves'])
    kifu_moves_len.append(moves_len)
    board = shogi.Board()
    max_hand_fu_in_game = 0
    for move in kifu['moves']:
        board.push_usi(move)
        max_hand_fu_in_game = max(max_hand_fu_in_game, board.pieces_in_hand[shogi.BLACK][shogi.PAWN], board.pieces_in_hand[shogi.WHITE][shogi.PAWN])
    max_hand_fu.append(max_hand_fu_in_game)

print('kifu num : {}'.format(kifu_num))
print('moves sum : {}'.format(sum(kifu_moves_len)))
print('moves mean : {}'.format(statistics.mean(kifu_moves_len)))
print('moves median : {}'.format(statistics.median(kifu_moves_len)))
print('moves max : {}'.format(max(kifu_moves_len)))
print('moves min : {}'.format(min(kifu_moves_len)))
print('max hand fu : {}'.format(max(max_hand_fu)))

plt.hist(kifu_moves_len)
plt.show()

plt.hist(max_hand_fu, bins=18, range=(0, 18))
plt.show()