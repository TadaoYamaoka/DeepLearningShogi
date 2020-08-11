import argparse
import os
import sys
import shogi
import shogi.KIF

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('moves', type=int)
args = parser.parse_args()

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

kifu_count = 0
game_count = {}
win_count = {}
win = { 'b' : 0, 'w' : 1 }
for filepath in find_all_files(args.dir):
    kif = shogi.KIF.Parser.parse_file(filepath)[0]
    if len(kif['moves']) <= args.moves:
        print(filepath)
        kifu_count = kifu_count + 1
        for i, name in enumerate(kif['names']):
            game_count[name] = game_count.get(name, 0) + 1
            if i == win[kif['win']]:
                win_count[name] = win_count.get(name, 0) + 1

print('kifu count :', kifu_count)
for name in game_count.keys():
    print('name : {}, games : {}, winrate : {:.4f}'.format(name, game_count[name], win_count.get(name, 0) / game_count[name]))
