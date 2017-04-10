import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

import shogi
import shogi.CSA

import argparse
import random
import copy

import logging
logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Deep Learning Shogi')
parser.add_argument('train_kifu_list', type=str, help='train kifu list')
parser.add_argument('test_kifu_list', type=str, help='test kifu list')
parser.add_argument('--batchsize', '-b', type=int, default=8, help='Number of positions in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
args = parser.parse_args()

k = 256
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Convolution2D(in_channels = None, out_channels = k, ksize = 3, pad = 1),
            l2=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l3=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l4=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l5=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l6=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l7=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l8=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l9=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l10=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l11=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l12=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1),
            l13=L.Convolution2D(in_channels = k, out_channels = len(shogi.PIECE_TYPES), ksize = 1, nobias = True),
            l13_2=L.Bias(shape=(9*9*len(shogi.PIECE_TYPES)))
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        h5 = F.relu(self.l5(h4))
        h6 = F.relu(self.l6(h5))
        h7 = F.relu(self.l7(h6))
        h8 = F.relu(self.l8(h7))
        h9 = F.relu(self.l9(h8))
        h10 = F.relu(self.l10(h9))
        h11 = F.relu(self.l11(h10))
        h12 = F.relu(self.l12(h11))
        h13 = self.l13(h12)
        return self.l13_2(F.reshape(h13, (len(h13.data), 9*9*len(shogi.PIECE_TYPES))))

model = MyChain()
model.to_gpu()

optimizer = optimizers.AdaGrad()
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
            if board.turn == shogi.BLACK:
                occupied = (board.occupied[shogi.BLACK], board.occupied[shogi.WHITE])
                pieces_in_hand = (board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE])
            else:
                occupied = (board.occupied[shogi.WHITE], board.occupied[shogi.BLACK])
                pieces_in_hand = (board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK])
            move_from = move[0:2]
            move_to = move[2:4]
            if move_from[1] == '*':
                move_piece = shogi.Piece.from_symbol(move_from[0]).piece_type
                if len(move) == 5 and move[4] == '+':
                    move_piece = shogi.PIECE_PROMOTED.index(move_piece)
            else:
                move_piece = board.piece_at(shogi.SQUARE_NAMES.index(move_from)).piece_type
            move_label = 9 * 9 * (move_piece - 1) + shogi.SQUARE_NAMES.index(move_to)
            positions.append(copy.deepcopy((board.piece_bb, occupied, pieces_in_hand, move_label)))
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
    piece_bb, occupied, pieces_in_hand, move = position
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
    # empty
    feature = np.zeros(9*9)
    for pos in shogi.SQUARES:
        if bb & shogi.BB_SQUARES[pos] == 0:
            feature[pos] = 1
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

    return (Variable(cuda.to_gpu(np.array(mini_batch_data, dtype=np.float32))), Variable(cuda.to_gpu(np.array(mini_batch_move, dtype=np.int32))))

def mini_batch_for_test(positions):
    mini_batch_data = []
    mini_batch_move = []
    for b in range(64):
        features, move = make_features(random.choice(positions))
        mini_batch_data.append(features)
        mini_batch_move.append(move)

    return (Variable(cuda.to_gpu(np.array(mini_batch_data, dtype=np.float32))), Variable(cuda.to_gpu(np.array(mini_batch_move, dtype=np.int32))))

# train
itr = 0
sum_loss = 0
eval_interval = 100
for e in range(args.epoch):
    positions_train_shuffled = random.sample(positions_train, len(positions_train))

    for i in range(0, len(positions_train_shuffled) - args.batchsize, args.batchsize):
        x, t = mini_batch(positions_train_shuffled, i)
        y = model(x)

        model.cleargrads()
        loss = F.softmax_cross_entropy(y, t)
        loss.backward()
        optimizer.update()

        itr += 1
        sum_loss += loss.data

        # eval test data
        if itr % eval_interval == 0:
            x, t = mini_batch_for_test(positions_test)
            y = model(x)
            logging.info('epoch = {}, iteration = {}, loss = {}, accuracy = {}'.format(e + 1, itr, sum_loss / eval_interval, F.accuracy(y, t).data))
            sum_loss = 0


print('save the model')
serializers.save_npz('model', model)
print('save the optimizer')
serializers.save_npz('state', optimizer)
