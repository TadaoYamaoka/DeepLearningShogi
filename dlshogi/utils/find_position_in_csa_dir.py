from cshogi import Board, move_to_usi
from cshogi import CSA
import os
import glob
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('sfen')
args = parser.parse_args()

ptn =re.compile(r'\s\d+$')

board_sfen = Board()
board_sfen.set_position(args.sfen)
sfen = ptn.sub("", board_sfen.sfen())

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True)

board = Board()
for filepath in csa_file_list:
    for kif in CSA.Parser.parse_file(filepath):
        try:
            board.set_sfen(kif.sfen)
            for i, move in enumerate(kif.moves):
                assert board.is_legal(move)
                if ptn.sub("", board.sfen()) == sfen:
                    print(filepath, i, move_to_usi(move))
                    break
                board.push(move)
        except:
            continue
