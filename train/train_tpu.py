# -*- coding:utf-8 -*-

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Activation, Flatten
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

import shogi
import shogi.CSA

import argparse
import random
import copy

import logging

parser = argparse.ArgumentParser(description='Deep Learning Shogi')
parser.add_argument('train_kifu_list', type=str, help='train kifu list')
parser.add_argument('test_kifu_list', type=str, help='test kifu list')
parser.add_argument('--batchsize', '-b', type=int, default=8, help='Number of positions in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--log', default=None, help='log file path')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

# move direction
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE,
    HAND
] = range(17)

MOVE_DIRECTION_PROMOTED = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT
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
    PROM_BISHOP_MOVE_DIRECTION_LABEL, PROM_ROOK_MOVE_DIRECTION_LABEL
]

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

# read all kifu
def read_kifu(kifu_list_file):
    f = open(kifu_list_file, 'r')
    positions = []
    for line in f.readlines():
        filepath = line.rstrip('\r\n')
        # kifu = shogi.CSA.Parser.parse_file(filepath, encoding='utf-8')[0]
        kifu = shogi.CSA.Parser.parse_file(filepath)[0]
        board = shogi.Board()
        for move in kifu['moves']:
            move_to = shogi.SQUARE_NAMES.index(move[2:4])
            if move[1] == '*':
                # in hand
                move_piece = shogi.Piece.from_symbol(move[0]).piece_type
                move_direction = HAND
            else:
                move_from = shogi.SQUARE_NAMES.index(move[0:2])
                move_piece = board.piece_at(shogi.SQUARE_NAMES.index(move[0:2])).piece_type

            if board.turn == shogi.BLACK:
                piece_bb = board.piece_bb
                occupied = (board.occupied[shogi.BLACK], board.occupied[shogi.WHITE])
                pieces_in_hand = (board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE])
            else:
                piece_bb = [bb_rotate_180(bb) for bb in board.piece_bb]
                occupied = (bb_rotate_180(board.occupied[shogi.WHITE]), bb_rotate_180(board.occupied[shogi.BLACK]))
                pieces_in_hand = (board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK])
                move_to = SQUARES_R180[move_to]
                if move[1] != '*':
                    move_from = SQUARES_R180[move_from]

            # move direction
            if move[1] != '*':
                to_x = move_to % 9
                to_y = int(move_to / 9)
                from_x = move_from % 9
                from_y = int(move_from / 9)
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
                if len(move) == 5 and move[4] == '+':
                    move_direction = MOVE_DIRECTION_PROMOTED[move_direction]

            move_direction_label = PIECE_MOVE_DIRECTION_LABEL[move_piece] + PIECE_MOVE_DIRECTION[move_piece].index(move_direction)
            move_label = 9 * 9 * move_direction_label + move_to
            positions.append(copy.deepcopy((piece_bb, occupied, pieces_in_hand, board.is_check(), move_label)))
            board.push_usi(move)
    f.close()
    return positions

logging.debug('read kifu start')
positions_train = read_kifu(args.train_kifu_list)
positions_test = read_kifu(args.test_kifu_list)
logging.debug('read kifu end')

logging.info('train position num = {}'.format(len(positions_train)))
logging.info('test position num = {}'.format(len(positions_test)))

def make_features(position):
    piece_bb, occupied, pieces_in_hand, is_check, move = position
    features = []
    for color in shogi.COLORS:
        # board pieces
        for piece_type in shogi.PIECE_TYPES_WITH_NONE[1:]:
            bb = piece_bb[piece_type] & occupied[color]
            feature = np.zeros(9*9)
            for pos in shogi.SQUARES:
                if bb & shogi.BB_SQUARES[pos] > 0:
                    feature[pos] = 1
            features.append(feature.reshape((9, 9)))

        # pieces in hand
        for piece_type in range(1, 8):
            for n in range(shogi.MAX_PIECES_IN_HAND[piece_type]):
                if piece_type in pieces_in_hand[color] and n < pieces_in_hand[color][piece_type]:
                    feature = np.ones(9*9)
                else:
                    feature = np.zeros(9*9)
                features.append(feature.reshape((9, 9)))

    # is check
    if is_check:
        feature = np.ones(9*9)
    else:
        feature = np.zeros(9*9)
    features.append(feature.reshape((9, 9)))

    return (features, move)

# mini batch
def mini_batch(positions, i):
    mini_batch_data = []
    mini_batch_move = []
    for b in range(args.batchsize):
        features, move = make_features(positions[i + b])
        mini_batch_data.append(features)
        mini_batch_move.append(move)

    return (np.array(mini_batch_data, dtype=np.float32),
            to_categorical(mini_batch_move, num_classes=MOVE_DIRECTION_LABEL_NUM*9*9))

# data generator
def datagen(positions):
    while True:
        positions_shuffled = random.sample(positions, len(positions))
        for i in range(0, len(positions_shuffled) - args.batchsize, args.batchsize):
            x, t = mini_batch(positions_shuffled, i)
            yield x, t

class Bias(Layer):

    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[1:]),
                                 initializer='uniform',
                                 trainable=True)
        super(Bias, self).build(input_shape)

    def call(self, x):
        return x + self.W

k = 256
model = Sequential()
# layer1
model.add(Conv2D(k, (3, 3), padding='same', data_format='channels_first', input_shape=((len(shogi.PIECE_TYPES) + sum(shogi.MAX_PIECES_IN_HAND))*2+1, 9, 9)))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
# layer2 - 12
for i in range(11):
    model.add(Conv2D(k, (3, 3), padding='same', data_format='channels_first'))
    if i < 8:
        model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
# layer13
model.add(Conv2D(MOVE_DIRECTION_LABEL_NUM, (1, 1), data_format='channels_first', use_bias=False))
model.add(Flatten())
model.add(Bias())
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

checkpoint = ModelCheckpoint('model-best.hdf5', verbose=1, save_best_only=True)

# TPU
import os
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import keras_support
tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

logging.info('Training start')
model.fit_generator(datagen(positions_train), int(len(positions_train) / args.batchsize),
          epochs=args.epoch,
          validation_data=datagen(positions_test), validation_steps=int(len(positions_test) / args.batchsize),
          callbacks=[checkpoint])
logging.info('Training end')

model.save('./model-final.hdf5', save_format="h5")

import gc; gc.collect()
logging.info('Done')
