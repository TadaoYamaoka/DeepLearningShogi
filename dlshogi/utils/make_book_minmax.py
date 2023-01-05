from email.policy import default
from cshogi import *
from cshogi import CSA
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('book')
parser.add_argument('--startsfen')
parser.add_argument('--th', type=int, default=100)
parser.add_argument('--uniq', action='store_true')
parser.add_argument('--side', choices=['black', 'white', 'both'], default='both')
parser.add_argument('--black_draw_value', type=float, default=0.4)
parser.add_argument('--white_draw_value', type=float, default=0.6)
args = parser.parse_args()

board = Board()
nodes = {}
kif_num = 0
duplicates = set()
for filepath in glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True):
    for kif in CSA.Parser.parse_file(filepath):
        endgame = kif.endgame
        if endgame not in ('%TORYO', '%SENNICHITE', '%KACHI'):
            continue
        # èdï°çÌèú
        if args.uniq:
            dup_key = ''.join([move_to_usi(move) for move in kif.moves])
            if dup_key in duplicates:
                # print(f'duplicate {filepath}')
                continue
            duplicates.add(dup_key)

        board.set_sfen(kif.sfen)
        try:
            for i, (move, score) in enumerate(zip(kif.moves, kif.scores)):
                assert board.is_legal(move)

                key = board.book_key()
                if key in nodes:
                    node = nodes[key]
                else:
                    node = { 'win': 0, 'draw': 0, 'num': 0, 'value': None, 'candidates': {} }
                    nodes[key] = node

                node['num'] += 1
                if board.turn == kif.win - 1:
                    node['win'] += 1
                elif kif.win == DRAW:
                    node['draw'] += 1

                assert abs(score) <= 1000000
                eval = min(32767, max(score, -32767))
                eval = eval if board.turn == BLACK else -eval

                candidates = node['candidates']
                if move in candidates:
                    candidate = candidates[move]
                else:
                    candidate = { 'sum_eval': 0, 'num': 0 }
                    candidates[move] = candidate

                candidate['sum_eval'] += eval
                candidate['num'] += 1

                board.push(move)
                if board.is_draw() == REPETITION_DRAW:
                    break
        except:
            print(f'skip {filepath}:{i}:{move_to_usi(move)}:{score}')
            continue

        kif_num += 1

print('kif num', kif_num)


def minmax(board):
    key = board.book_key()
    assert key in nodes

    node = nodes[key]
    if node['num'] < args.th:
        return None

    if board.is_draw() == REPETITION_DRAW:
        return None

    max_value = -10000
    for move, candidate in node['candidates'].items():
        board.push(move)
        value = minmax(board)
        board.pop()

        if value is not None:
            max_value = max(max_value, 1 - value)

    if max_value == -10000:
        draw_value = args.black_draw_value if board.turn == BLACK else args.white_draw_value
        max_value = (node['win'] + node['draw'] * draw_value) / node['num']
        # print(board.sfen(), max_value)

    node['value'] = max_value
    return max_value

if args.startsfen:
    startsfen = args.startsfen
else:
    startsfen = STARTING_SFEN

board.set_sfen(startsfen)

minmax(board)

# íËê’èoóÕ
if args.side == 'black':
    side = BLACK
elif args.side == 'white':
    side = WHITE
def make_book(board, book):
    key = board.book_key()

    node = nodes[key]
    if node['value'] is None:
        return None, None

    if board.is_draw() == REPETITION_DRAW:
        return node['value'], None

    candidates = []
    for move, candidate in node['candidates'].items():
        board.push(move)
        value, next_move = make_book(board, book)
        if value is not None:
            candidates.append({ 'move': move, 'next_move': next_move, 'value': 1 - value, 'eval': int(candidate['sum_eval'] / candidate['num']) })
        board.pop()

    if args.side == 'both' or board.turn == side:
        candidates.sort(key=lambda x: x['value'], reverse=True)
        book[key] = { 'sfen': board.sfen(), 'candidates': candidates }

    return node['value'], candidates[0]['move'] if len(candidates) > 0 else None

book = {}
make_book(board, book)

print('book position num', len(book))
with open(args.book, 'w') as f:
    f.write('#YANEURAOU-DB2016 1.00\n')
    for key in sorted(book.keys()):
        entry = book[key]
        if len(entry['candidates']) == 0:
            continue

        f.write('sfen ' + entry['sfen'] + '\n')

        for candidate in entry['candidates']:
            next_move = move_to_usi(candidate['next_move']) if candidate['next_move'] is not None else 'none'
            f.write(f"{move_to_usi(candidate['move'])} {next_move} {candidate['eval']} 0 {int(candidate['value'] * 100)}\n")
