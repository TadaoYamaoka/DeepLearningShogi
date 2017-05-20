import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import Chain
import chainer.functions as F
import chainer.links as L

import shogi
import shogi.CSA

from dlshogi.common import *

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

k = 192
w = 3
dropout_ratio = 0.1
class PolicyNetwork(Chain):
    def __init__(self):
        super(PolicyNetwork, self).__init__(
            l1_1=L.Convolution2D(in_channels = None, out_channels = k, ksize = w, pad = int(w/2), nobias = True),
            l1_2=L.Convolution2D(in_channels = None, out_channels = k, ksize = 1, nobias = True), # pieces_in_hand
            l2=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l3=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l4=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l5=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l6=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l7=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l8=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l9=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l10=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l11=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l12=L.Convolution2D(in_channels = k, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1, nobias = True),
            l12_2=L.Bias(shape=(9*9*MOVE_DIRECTION_LABEL_NUM)),
            norm1=L.BatchNormalization(k),
            norm2=L.BatchNormalization(k),
            norm3=L.BatchNormalization(k),
            norm4=L.BatchNormalization(k),
            norm5=L.BatchNormalization(k),
            norm6=L.BatchNormalization(k),
            norm7=L.BatchNormalization(k),
            norm8=L.BatchNormalization(k),
            norm9=L.BatchNormalization(k),
            norm10=L.BatchNormalization(k)
        )

    def __call__(self, x1, x2, test=False):
        u1_1 = self.l1_1(x1)
        u1_2 = self.l1_2(x2)
        u1 = u1_1 + u1_2
        # Residual block
        h1 = F.relu(self.norm1(u1, test))
        h2 = F.dropout(F.relu(self.norm2(self.l2(h1), test)), ratio=dropout_ratio, train=not test)
        u3 = self.l3(h2) + u1
        # Residual block
        h3 = F.relu(self.norm3(u3, test))
        h4 = F.dropout(F.relu(self.norm4(self.l4(h3), test)), ratio=dropout_ratio, train=not test)
        u5 = self.l5(h4) + u3
        # Residual block
        h5 = F.relu(self.norm5(u5, test))
        h6 = F.dropout(F.relu(self.norm6(self.l6(h5), test)), ratio=dropout_ratio, train=not test)
        u7 = self.l7(h6) + u5
        # Residual block
        h7 = F.relu(self.norm7(u7, test))
        h8 = F.dropout(F.relu(self.norm8(self.l8(h7), test)), ratio=dropout_ratio, train=not test)
        u9 = self.l9(h8) + u7
        # Residual block
        h9 = F.relu(self.norm9(u9, test))
        h10 = F.dropout(F.relu(self.norm10(self.l10(h9), test)), ratio=dropout_ratio, train=not test)
        u11 = self.l11(h10) + u9
        # output
        h12 = self.l12(u11)
        return self.l12_2(F.reshape(h12, (len(h12.data), 9*9*MOVE_DIRECTION_LABEL_NUM)))

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
            for n in range(shogi.MAX_PIECES_IN_HAND[piece_type]):
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

def make_features(position):
    piece_bb, occupied, pieces_in_hand, is_check, move = position
    features1, features2 = make_input_features(piece_bb, occupied, pieces_in_hand, is_check)

    return (features1, features2, move)

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

def make_output_label(board, move):
    move_to = move.to_square
    move_from = move.from_square
    if move_from is None:
        # in hand
        move_piece = move.drop_piece_type
        move_direction = HAND
    else:
        move_piece = board.piece_at(move.from_square).piece_type

    if board.turn == shogi.WHITE:
        move_to = SQUARES_R180[move_to]
        if move_from is not None:
            move_from = SQUARES_R180[move_from]

    # move direction
    if move_from is not None:
        to_y, to_x = divmod(move_to, 9)
        from_y, from_x = divmod(move_from, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y
        if dir_y < 0 and dir_x == 0:
            move_direction = UP
        elif dir_y < 0 and dir_x < 0:
            move_direction = UP_LEFT
        elif dir_y < 0 and dir_x > 0:
            move_direction = UP_RIGHT
        elif dir_y == 0 and dir_x < 0:
            move_direction = LEFT
        elif dir_y == 0 and dir_x > 0:
            move_direction = RIGHT
        elif dir_y > 0 and dir_x == 0:
            move_direction = DOWN
        elif dir_y > 0 and dir_x < 0:
            move_direction = DOWN_LEFT
        elif dir_y > 0 and dir_x > 0:
            move_direction = DOWN_RIGHT

        # promote
        if move.promotion:
            move_direction = MOVE_DIRECTION_PROMOTED[move_direction]

    move_direction_label = PIECE_MOVE_DIRECTION_LABEL[move_piece] + PIECE_MOVE_DIRECTION[move_piece].index(move_direction)
    move_label = 9 * 9 * move_direction_label + move_to

    return move_label
