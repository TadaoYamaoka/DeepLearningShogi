import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from dlshogi.policy_network import *
from dlshogi.common import *

import shogi
import shogi.CSA

import argparse
import random
import copy

import logging

parser = argparse.ArgumentParser(description='Deep Learning Shogi')
parser.add_argument('train_kifu_list', type=str, help='train kifu list')
parser.add_argument('test_kifu_list', type=str, help='test kifu list')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of positions in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--model', type=str, default='model', help='model file name')
parser.add_argument('--state', type=str, default='state', help='state file name')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

model = PolicyNetwork()
model.to_gpu()

optimizer = optimizers.SGD(lr=args.lr)
optimizer.use_cleargrads()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

# read all kifu
def read_kifu(kifu_list_file):
    f = open(kifu_list_file, 'r')
    positions = []
    for line in f.readlines():
        filepath = line.rstrip('\r\n')
        kifu = shogi.CSA.Parser.parse_file(filepath, encoding='utf-8')[0]
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

# mini batch
def mini_batch(positions, i):
    mini_batch_data1 = []
    mini_batch_data2 = []
    mini_batch_move = []
    for b in range(args.batchsize):
        features1, features2, move = make_features(positions[i + b])
        mini_batch_data1.append(features1)
        mini_batch_data2.append(features2)
        mini_batch_move.append(move)

    return (Variable(cuda.to_gpu(np.array(mini_batch_data1, dtype=np.float32))), Variable(cuda.to_gpu(np.array(mini_batch_data2, dtype=np.float32))), Variable(cuda.to_gpu(np.array(mini_batch_move, dtype=np.int32))))

def mini_batch_for_test(positions):
    mini_batch_data1 = []
    mini_batch_data2 = []
    mini_batch_move = []
    for b in range(640):
        features1, features2, move = make_features(random.choice(positions))
        mini_batch_data1.append(features1)
        mini_batch_data2.append(features2)
        mini_batch_move.append(move)

    return (Variable(cuda.to_gpu(np.array(mini_batch_data1, dtype=np.float32))), Variable(cuda.to_gpu(np.array(mini_batch_data2, dtype=np.float32))), Variable(cuda.to_gpu(np.array(mini_batch_move, dtype=np.int32))))

# train
itr = 0
sum_loss = 0
eval_interval = 1000
for e in range(args.epoch):
    positions_train_shuffled = random.sample(positions_train, len(positions_train))

    itr_epoch = 0
    sum_loss_epoch = 0
    for i in range(0, len(positions_train_shuffled) - args.batchsize, args.batchsize):
        x1, x2, t = mini_batch(positions_train_shuffled, i)
        y = model(x1, x2)

        model.cleargrads()
        loss = F.softmax_cross_entropy(y, t)
        loss.backward()
        optimizer.update()

        itr += 1
        sum_loss += loss.data
        itr_epoch += 1
        sum_loss_epoch += loss.data

        # print train loss and test accuracy
        if optimizer.t % eval_interval == 0:
            x1, x2, t = mini_batch_for_test(positions_test)
            y = model(x1, x2, test=True)
            logging.info('epoch = {}, iteration = {}, loss = {}, accuracy = {}'.format(optimizer.epoch + 1, optimizer.t, sum_loss / itr, F.accuracy(y, t).data))
            itr = 0
            sum_loss = 0

    # validate test data
    itr_test = 0
    sum_test_accuracy = 0
    for i in range(0, len(positions_test) - args.batchsize, args.batchsize):
        x1, x2, t = mini_batch(positions_test, i)
        y = model(x1, x2, test=True)
        itr_test += 1
        sum_test_accuracy += F.accuracy(y, t).data
    logging.info('epoch = {}, iteration = {}, train loss avr = {}, test accuracy = {}'.format(optimizer.epoch + 1, optimizer.t, sum_loss_epoch / itr_epoch, sum_test_accuracy / itr_test))
    
    optimizer.new_epoch()

print('save the model')
serializers.save_npz(args.model, model)
print('save the optimizer')
serializers.save_npz(args.state, optimizer)
