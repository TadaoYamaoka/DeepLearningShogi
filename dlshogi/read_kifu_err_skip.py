import shogi
import shogi.CSA
import copy

from dlshogi.features import *

import numpy as np
from cshogi import *

# read kifu
def read_kifu(kifu_list_file):
    positions = []
    with open(kifu_list_file, 'r') as f:
        for line in f.readlines():
            try:
                filepath = line.rstrip('\r\n')
                kifu = shogi.CSA.Parser.parse_file(filepath)[0]
                win_color = shogi.BLACK if kifu['win'] == 'b' else shogi.WHITE
                board = shogi.Board()
                for move in kifu['moves']:
                    if board.turn == shogi.BLACK:
                        piece_bb = copy.deepcopy(board.piece_bb)
                        occupied = copy.deepcopy((board.occupied[shogi.BLACK], board.occupied[shogi.WHITE]))
                        pieces_in_hand = copy.deepcopy((board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE]))
                    else:
                        piece_bb = [bb_rotate_180(bb) for bb in board.piece_bb]
                        occupied = (bb_rotate_180(board.occupied[shogi.WHITE]), bb_rotate_180(board.occupied[shogi.BLACK]))
                        pieces_in_hand = copy.deepcopy((board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK]))

                    # move label
                    move_label = make_output_label(shogi.Move.from_usi(move), board.turn)

                    # result
                    win = 1 if win_color == board.turn else 0

                    positions.append((piece_bb, occupied, pieces_in_hand, move_label, win))
                    board.push_usi(move)
            except:
                print("skip -> " + filepath)
                pass
    return positions

def read_kifu_from_hcpe(hcpe_path):
    hcpes = np.fromfile(hcpe_path, dtype=HuffmanCodedPosAndEval)
    positions = []
    for i, hcpe in enumerate(hcpes):
        board = Board()        
        board.set_hcp(hcpes[i]['hcp'])
        
        sfen = board.sfen()
        pboard = shogi.Board(sfen=sfen.decode('utf-8'))
        if pboard.turn == shogi.BLACK:
            piece_bb = copy.deepcopy(pboard.piece_bb)
            occupied = copy.deepcopy((pboard.occupied[shogi.BLACK], pboard.occupied[shogi.WHITE]))
            pieces_in_hand = copy.deepcopy((pboard.pieces_in_hand[shogi.BLACK], pboard.pieces_in_hand[shogi.WHITE]))
        else:
            piece_bb = [bb_rotate_180(bb) for bb in pboard.piece_bb]
            occupied = (bb_rotate_180(pboard.occupied[shogi.WHITE]), bb_rotate_180(pboard.occupied[shogi.BLACK]))
            pieces_in_hand = copy.deepcopy((pboard.pieces_in_hand[shogi.WHITE], pboard.pieces_in_hand[shogi.BLACK]))
        
        # move label
        move = hcpes[i]['bestMove16']
        move_label = make_output_label(shogi.Move.from_usi(move_to_usi(move).decode('utf-8')), pboard.turn)
        
        # result
        gameResult = hcpes[i]['gameResult']
        if board.turn == BLACK:
            if gameResult == BLACK_WIN:
                win_color = 1
            if gameResult == WHITE_WIN:
                win_color = -1
            else:
                win_color = 0
        else:
            if gameResult == BLACK_WIN:
                win_color = -1
            if gameResult == WHITE_WIN:
                win_color = 1
            else:
                win_color = 0

        win = 1 if win_color == pboard.turn else 0
        
        positions.append((piece_bb, occupied, pieces_in_hand, move_label, win))
        
    return positions
