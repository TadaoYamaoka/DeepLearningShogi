import argparse
import os
import sys
import shogi.KIF

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
args = parser.parse_args()

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

win = {}
player = {}
for filepath in find_all_files(args.dir):
    summary = shogi.KIF.Parser.parse_file(filepath)[0]
    winner = summary['win']
    win[winner] = win.get(winner, 0) + 1

    names = summary['names']
    for i in range(2):
        if names[i] not in player:
            player[names[i]] = {}

        if i == 0 and winner == 'b' or i == 1 and winner == 'w':
            # 勝ち
            player[names[i]]['win'] = player[names[i]].get('win', 0) + 1
            # 先手勝ち後手勝ち
            player[names[i]][winner] = player[names[i]].get(winner, 0) + 1

# 表示
games = win['b'] + win['w'] + win.get('-', 0)
print('対局数{} 先手勝ち{}({}%) 後手勝ち{}({}%) 引き分け{}'.format(
    games, win['b'], int(win['b'] / games * 100), win['w'], int(win['w'] / games * 100), win.get('-', 0)))
for name in player.keys():
    print(name)
    print('勝ち{}({}%) 先手勝ち{}({}%) 後手勝ち{}({}%)'.format(
        player[name]['win'], int(player[name]['win'] / games * 100),
        player[name]['b'], int(player[name]['b'] / games * 100),
        player[name]['w'], int(player[name]['w'] / games * 100)))
