import numpy as np
from cshogi import *
import glob
import os.path

MAX_MOVE_COUNT = 512

# process csa
def process_csa(f, csa_file_list, filter_moves, filter_rating, limit_moves):
    board = Board()
    parser = Parser()
    num_games = 0
    num_positions = 0
    hcps = np.empty(MAX_MOVE_COUNT, dtype=HuffmanCodedPos)
    for filepath in csa_file_list:
        parser.parse_csa_file(filepath)
        if parser.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI', '%HIKIWAKE') or len(parser.moves) < filter_moves:
            continue
        if filter_rating > 0 and (parser.ratings[0] < filter_rating or parser.ratings[1] < filter_rating):
            continue
        board.set_sfen(parser.sfen)
        assert board.is_ok(), "{}:{}".format(filepath, parser.sfen)
        # gameResult
        skip = False
        kachi = parser.endgame == '%KACHI'
        if args.before_kachi_moves:
            kachi_moves = len(parser.moves) - args.before_kachi_moves
        else:
            kachi_moves = None
        for i, (move, score) in enumerate(zip(parser.moves, parser.scores)):
            if not board.is_legal(move):
                print("skip {}:{}:{}".format(filepath, i, move_to_usi(move)))
                skip = True
                break

            # hcp
            if i >= limit_moves:
                if kachi_moves and kachi and i <= kachi_moves:
                    board.to_hcp(np.asarray(hcps[i]))
                else:
                    break
            else:
                board.to_hcp(np.asarray(hcps[i]))

            board.push(move)

        if skip:
            continue

        # write data
        hcps[:i+1].tofile(f)

        num_positions += i+1
        num_games += 1

    return num_games, num_positions

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('csa_dir', help='directory stored CSA file')
    parser.add_argument('hcp', help='hcp file')
    parser.add_argument('--filter_moves', type=int, default=80, help='filter by move count')
    parser.add_argument('--filter_rating', type=int, default=0, help='filter by rating')
    parser.add_argument('--recursive', '-r', action='store_true')
    parser.add_argument('--limit_moves', type=int, default=40, help='upper limit of move count')
    parser.add_argument('--before_kachi_moves', type=int, help='output the position until N moves before sengen kachi')

    args = parser.parse_args()

    if args.recursive:
        dir = os.path.join(args.csa_dir, '**')
    else:
        dir = args.csa_dir
    csa_file_list = glob.glob(os.path.join(dir, '*.csa'), recursive=args.recursive)

    with open(args.hcp, 'wb') as f:
        num_games, num_positions = process_csa(f, csa_file_list, args.filter_moves, args.filter_rating, args.limit_moves)
        print(f"games : {num_games}")
        print(f"positions : {num_positions}")
