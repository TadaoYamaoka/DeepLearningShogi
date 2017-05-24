import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from dlshogi.value_network import *
from dlshogi.common import *
from dlshogi.teacher_data import *

import shogi
import shogi.CSA

import argparse
import random
import copy
import os

import logging

parser = argparse.ArgumentParser(description='Train value network')
parser.add_argument('train_data', type=str, help='train data file')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of positions in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--model', type=str, default='model_val', help='model file name')
parser.add_argument('--state', type=str, default='state_val', help='state file name')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

model = ValueNetwork()
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

# read teacher data
def read_teacher_data(path):
    hcpevec = []
    with open(path, 'rb') as f:
        filesize = os.fstat(f.fileno()).st_size
        while f.tell() < filesize:
            hcpe = HuffmanCodedPosAndEval()
            f.readinto(hcpe)
            hcpevec.append(hcpe)
    return hcpevec

logging.debug('read teacher data start')
train_data = read_teacher_data(args.train_data)
test_data = read_teacher_data(args.test_data)
logging.debug('read teacher data end')

logging.info('train position num = {}'.format(len(train_data)))
logging.info('test position num = {}'.format(len(test_data)))

# mini batch
def mini_batch(hcpevec, i):
    mini_batch_data1 = []
    mini_batch_data2 = []
    mini_batch_val = []
    for b in range(args.batchsize):
        board, eval, bestMove, win = decode_hcpe(hcpevec[i + b])
        features1, features2 = make_input_features_from_board(board)
        mini_batch_data1.append(features1)
        mini_batch_data2.append(features2)
        mini_batch_val.append(win)

    return (Variable(cuda.to_gpu(np.array(mini_batch_data1, dtype=np.float32))),
            Variable(cuda.to_gpu(np.array(mini_batch_data2, dtype=np.float32))),
            Variable(cuda.to_gpu(np.array(mini_batch_val, dtype=np.float32).reshape((len(mini_batch_val), 1))))
            )

# train
itr = 0
sum_loss = 0
eval_interval = 1000
for e in range(args.epoch):
    train_data_shuffled = random.sample(train_data, len(train_data))

    itr_epoch = 0
    sum_loss_epoch = 0
    for i in range(0, len(train_data_shuffled) - args.batchsize, args.batchsize):
        x1, x2, t = mini_batch(train_data_shuffled, i)
        y = model(x1, x2)

        model.cleargrads()
        loss = F.mean_squared_error(y, t)
        loss.backward()
        optimizer.update()

        itr += 1
        sum_loss += loss.data
        itr_epoch += 1
        sum_loss_epoch += loss.data

        # print train loss and test accuracy
        if optimizer.t % eval_interval == 0:
            logging.info('epoch = {}, iteration = {}, loss = {}'.format(optimizer.epoch + 1, optimizer.t, sum_loss / itr))
            itr = 0
            sum_loss = 0

    # validate test data
    itr_test = 0
    sum_test_loss = 0
    for i in range(0, len(test_data) - args.batchsize, args.batchsize):
        x1, x2, t = mini_batch(test_data, i)
        y = model(x1, x2, test=True)
        itr_test += 1
        sum_test_loss += F.mean_squared_error(y, t).data
    logging.info('epoch = {}, iteration = {}, train loss avr = {}, test loss = {}'.format(optimizer.epoch + 1, optimizer.t, sum_loss_epoch / itr_epoch, sum_test_loss / itr_test))
    
    optimizer.new_epoch()

print('save the model')
serializers.save_npz(args.model, model)
print('save the optimizer')
serializers.save_npz(args.state, optimizer)
