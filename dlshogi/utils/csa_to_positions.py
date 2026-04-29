from cshogi import *
from cshogi import CSA
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('positions')
parser.add_argument('--limit_moves', type=int)
parser.add_argument('--limit_last_moves', type=int)
parser.add_argument('--limit_score', type=int)
parser.add_argument('--filter_rating', type=int)
args = parser.parse_args()

csa_file_list = glob.glob(os.path.join(args.dir, '**', '*.csa'), recursive=True)

board = Board()
parser = CSA.Parser()
num_games = 0
positions = set()
for filepath in csa_file_list:
    try:
        parser.parse_csa_file(filepath)
        if args.filter_rating:
            if parser.win == DRAW:
               if parser.ratings[0] < args.filter_rating or parser.ratings[1] < args.filter_rating:
                continue
            elif parser.ratings[2 - parser.win] < args.filter_rating:
                continue
        board.set_sfen(parser.sfen)
        assert board.is_ok(), "{}:{}".format(filepath, parser.sfen)
        prev_score = 0
        for i, (move, score) in enumerate(zip(parser.moves, parser.scores)):
            if args.limit_moves and i >= args.limit_moves:
                break
            
            if args.limit_last_moves and i >= len(parser.moves) - args.limit_last_moves:
                break
            
            if args.limit_score and abs(score) > args.limit_score and abs(prev_score) > args.limit_score:
                break
            prev_score = score

            if not board.is_legal(move):
                print("skip {}:{}:{}".format(filepath, i, move_to_usi(move)))
                break

            board.push(move)

        position = 'startpos moves ' + ' '.join([move_to_usi(move) for move in board.history])
        positions.add(position)

        num_games += 1
    except Exception as e:
        print("skip {} {}".format(filepath, e))

print(f"games : {num_games}")

# 保存
with open(args.positions, 'w') as f:
    for position in positions:
        f.write(position + '\n')
