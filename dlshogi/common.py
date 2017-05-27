import numpy as np
import shogi

# rotate 180degree
SQUARES_R180 = [
    shogi.I1, shogi.I2, shogi.I3, shogi.I4, shogi.I5, shogi.I6, shogi.I7, shogi.I8, shogi.I9,
    shogi.H1, shogi.H2, shogi.H3, shogi.H4, shogi.H5, shogi.H6, shogi.H7, shogi.H8, shogi.H9,
    shogi.G1, shogi.G2, shogi.G3, shogi.G4, shogi.G5, shogi.G6, shogi.G7, shogi.G8, shogi.G9,
    shogi.F1, shogi.F2, shogi.F3, shogi.F4, shogi.F5, shogi.F6, shogi.F7, shogi.F8, shogi.F9,
    shogi.E1, shogi.E2, shogi.E3, shogi.E4, shogi.E5, shogi.E6, shogi.E7, shogi.E8, shogi.E9,
    shogi.D1, shogi.D2, shogi.D3, shogi.D4, shogi.D5, shogi.D6, shogi.D7, shogi.D8, shogi.D9,
    shogi.C1, shogi.C2, shogi.C3, shogi.C4, shogi.C5, shogi.C6, shogi.C7, shogi.C8, shogi.C9,
    shogi.B1, shogi.B2, shogi.B3, shogi.B4, shogi.B5, shogi.B6, shogi.B7, shogi.B8, shogi.B9,
    shogi.A1, shogi.A2, shogi.A3, shogi.A4, shogi.A5, shogi.A6, shogi.A7, shogi.A8, shogi.A9,
]
def bb_rotate_180(bb):
    bb_r180 = 0
    for pos in shogi.SQUARES:
        if bb & shogi.BB_SQUARES[pos] > 0:
            bb_r180 += 1 << SQUARES_R180[pos]
    return bb_r180

# move direction
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE,
    HAND
] = range(17)

MOVE_DIRECTION_PROMOTED = [
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE
]

PAWN_MOVE_DIRECTION = [UP, UP_PROMOTE, HAND]
LANCE_MOVE_DIRECTION = [UP, UP_PROMOTE, HAND]
KNIGHT_MOVE_DIRECTION = [UP_LEFT, UP_RIGHT,
                         UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE,
                         HAND]
SILVER_MOVE_DIRECTION = [UP, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT,
                         UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE,
                         HAND]
GOLD_MOVE_DIRECTION = [UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN,
                       UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN,
                       HAND]
BISHOP_MOVE_DIRECTION = [UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT,
                         UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE,
                         HAND]
ROOK_MOVE_DIRECTION = [UP, LEFT, RIGHT, DOWN,
                       UP_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE,
                       HAND]
KING_MOVE_DIRECTION = [UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT]
PROM_PAWN_MOVE_DIRECTION = [UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN]
PROM_LANCE_MOVE_DIRECTION = [UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN]
PROM_KNIGHT_MOVE_DIRECTION = [UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN]
PROM_SILVER_MOVE_DIRECTION = [UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN]
PROM_BISHOP_MOVE_DIRECTION = [UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT]
PROM_ROOK_MOVE_DIRECTION = [UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT]

PIECE_MOVE_DIRECTION = [
    None,
    PAWN_MOVE_DIRECTION, LANCE_MOVE_DIRECTION, KNIGHT_MOVE_DIRECTION, SILVER_MOVE_DIRECTION,
    GOLD_MOVE_DIRECTION,
    BISHOP_MOVE_DIRECTION, ROOK_MOVE_DIRECTION,
    KING_MOVE_DIRECTION,
    PROM_PAWN_MOVE_DIRECTION, PROM_LANCE_MOVE_DIRECTION, PROM_KNIGHT_MOVE_DIRECTION, PROM_SILVER_MOVE_DIRECTION,
    PROM_BISHOP_MOVE_DIRECTION, PROM_ROOK_MOVE_DIRECTION
]

# classification label
PAWN_MOVE_DIRECTION_LABEL = 0
LANCE_MOVE_DIRECTION_LABEL = len(PAWN_MOVE_DIRECTION)
KNIGHT_MOVE_DIRECTION_LABEL = LANCE_MOVE_DIRECTION_LABEL + len(LANCE_MOVE_DIRECTION)
SILVER_MOVE_DIRECTION_LABEL = KNIGHT_MOVE_DIRECTION_LABEL + len(KNIGHT_MOVE_DIRECTION)
GOLD_MOVE_DIRECTION_LABEL = SILVER_MOVE_DIRECTION_LABEL + len(SILVER_MOVE_DIRECTION)
BISHOP_MOVE_DIRECTION_LABEL = GOLD_MOVE_DIRECTION_LABEL + len(GOLD_MOVE_DIRECTION)
ROOK_MOVE_DIRECTION_LABEL = BISHOP_MOVE_DIRECTION_LABEL + len(BISHOP_MOVE_DIRECTION)
KING_MOVE_DIRECTION_LABEL = ROOK_MOVE_DIRECTION_LABEL + len(ROOK_MOVE_DIRECTION)
PROM_PAWN_MOVE_DIRECTION_LABEL = KING_MOVE_DIRECTION_LABEL + len(KING_MOVE_DIRECTION)
PROM_LANCE_MOVE_DIRECTION_LABEL = PROM_PAWN_MOVE_DIRECTION_LABEL + len(PROM_PAWN_MOVE_DIRECTION)
PROM_KNIGHT_MOVE_DIRECTION_LABEL = PROM_LANCE_MOVE_DIRECTION_LABEL + len(PROM_LANCE_MOVE_DIRECTION)
PROM_SILVER_MOVE_DIRECTION_LABEL = PROM_KNIGHT_MOVE_DIRECTION_LABEL + len(PROM_KNIGHT_MOVE_DIRECTION)
PROM_BISHOP_MOVE_DIRECTION_LABEL = PROM_SILVER_MOVE_DIRECTION_LABEL + len(PROM_SILVER_MOVE_DIRECTION)
PROM_ROOK_MOVE_DIRECTION_LABEL = PROM_BISHOP_MOVE_DIRECTION_LABEL + len(PROM_BISHOP_MOVE_DIRECTION)
MOVE_DIRECTION_LABEL_NUM = PROM_ROOK_MOVE_DIRECTION_LABEL + len(PROM_ROOK_MOVE_DIRECTION)

PIECE_MOVE_DIRECTION_LABEL = [
    None,
    PAWN_MOVE_DIRECTION_LABEL, LANCE_MOVE_DIRECTION_LABEL, KNIGHT_MOVE_DIRECTION_LABEL, SILVER_MOVE_DIRECTION_LABEL,
    GOLD_MOVE_DIRECTION_LABEL,
    BISHOP_MOVE_DIRECTION_LABEL, ROOK_MOVE_DIRECTION_LABEL,
    KING_MOVE_DIRECTION_LABEL,
    PROM_PAWN_MOVE_DIRECTION_LABEL, PROM_LANCE_MOVE_DIRECTION_LABEL, PROM_KNIGHT_MOVE_DIRECTION_LABEL, PROM_SILVER_MOVE_DIRECTION_LABEL,
    PROM_BISHOP_MOVE_DIRECTION_LABEL, PROM_ROOK_MOVE_DIRECTION_LABEL,
    MOVE_DIRECTION_LABEL_NUM
]

MAX_PIECES_IN_HAND = list(shogi.MAX_PIECES_IN_HAND)
MAX_PIECES_IN_HAND[shogi.PAWN] = 8 # 歩の持ち駒の上限

def make_input_features(piece_bb, occupied, pieces_in_hand, is_check):
    features1 = []
    features2 = []
    for color in shogi.COLORS:
        # board pieces
        for piece_type in shogi.PIECE_TYPES_WITH_NONE[1:]:
            bb = piece_bb[piece_type] & occupied[color]
            feature = np.zeros(9*9)
            for pos in shogi.SQUARES:
                if bb & shogi.BB_SQUARES[pos] > 0:
                    feature[pos] = 1
            features1.append(feature.reshape((9, 9)))

        # pieces in hand
        for piece_type in range(1, 8):
            for n in range(MAX_PIECES_IN_HAND[piece_type]):
                if piece_type in pieces_in_hand[color] and n < pieces_in_hand[color][piece_type]:
                    feature = np.ones(9*9)
                else:
                    feature = np.zeros(9*9)
                features2.append(feature.reshape((9, 9)))

    # is check
    if is_check:
        feature = np.ones(9*9)
    else:
        feature = np.zeros(9*9)
    features2.append(feature.reshape((9, 9)))

    return features1, features2

def make_input_features_from_board(board):
    if board.turn == shogi.BLACK:
        piece_bb = board.piece_bb
        occupied = (board.occupied[shogi.BLACK], board.occupied[shogi.WHITE])
        pieces_in_hand = (board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE])
    else:
        piece_bb = [bb_rotate_180(bb) for bb in board.piece_bb]
        occupied = (bb_rotate_180(board.occupied[shogi.WHITE]), bb_rotate_180(board.occupied[shogi.BLACK]))
        pieces_in_hand = (board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK])
    return make_input_features(piece_bb, occupied, pieces_in_hand, board.is_check())
