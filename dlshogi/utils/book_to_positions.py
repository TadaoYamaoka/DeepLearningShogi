import argparse
import numpy as np
from collections import namedtuple
from cshogi import *

parser = argparse.ArgumentParser()
parser.add_argument('book')
parser.add_argument('positions')
parser.add_argument('--depth', type=int, default=512)
args = parser.parse_args()

book = np.fromfile(args.book, BookEntry)
keys = set(np.unique(book['key']))
exists = set()

def get_hcp(board):
    hcp = np.empty(1, HuffmanCodedPos)
    board.to_hcp(hcp)
    return hcp

# 1手先に登録されているか
def exist_next(board):
    for move in board.legal_moves:
        if board.book_key_after(key, move) in keys:
            return True
    return False

board = Board()
PositionWithMove = namedtuple('PositionWithMove', 'hcp move parent')
current_positions = [PositionWithMove(get_hcp(board), None, None)]
depth = 0
positions = []

while len(current_positions) > 0 and depth < args.depth:
    next_positions = []
    for current_position in current_positions:
        board.set_hcp(np.asarray(current_position.hcp))
        for move in board.legal_moves:
            board.push(move)
            key = board.book_key()
            if key in exists:
                # 合流
                board.pop()
                continue
            exists.add(key)

            if key in keys or exist_next(board):
                next_position = PositionWithMove(get_hcp(board), move, current_position)
                if key in keys:
                    positions.append(next_position)
                next_positions.append(next_position)
            board.pop()
    current_positions = next_positions
    depth += 1

with open(args.positions, 'w') as f:
    for position in positions:
        moves = []
        while position.parent is not None:
            moves.append(position.move)
            position = position.parent
        f.write('startpos moves ' + ' '.join([move_to_usi(move) for move in reversed(moves)]) + '\n')

print('positions', len(positions))
print('depth', depth)
