import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from softmax_cross_entropy_with_weight import *

from dlshogi.policy_network import *
from dlshogi.common import *

import cppshogi

import argparse
import random

import logging

parser = argparse.ArgumentParser(description='Traning RL policy network using hcpe')
parser.add_argument('train_data', type=str, help='train data file')
parser.add_argument('--batchsize', '-b', type=int, default=64, help='Number of positions in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--model', type=str, default='model_rl_hcpe', help='model file name')
parser.add_argument('--state', type=str, default='state_rl_hcpe', help='state file name')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

model = PolicyNetwork()
model.to_gpu()

alpha = args.lr
optimizer = optimizers.SGD(lr=alpha)
optimizer.use_cleargrads()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

logging.debug('read teacher data start')
train_data = np.fromfile(args.train_data, dtype=HuffmanCodedPosAndEval)
logging.debug('read teacher data end')

logging.info('train position num = {}'.format(len(train_data)))

# mini batch
def mini_batch(hcpevec):
    features1 = np.empty((len(hcpevec), 2 * 14, 9, 9), dtype=np.float32)
    features2 = np.empty((len(hcpevec), 2 * MAX_PIECES_IN_HAND_SUM + 1, 9, 9), dtype=np.float32)
    move = np.empty((len(hcpevec)), dtype=np.int32)
    result = np.empty((len(hcpevec)), dtype=np.float32)
    value = np.empty((len(hcpevec)), dtype=np.float32)

    cppshogi.hcpe_decode_with_value(hcpevec, features1, features2, move, result, value)

    return (Variable(cuda.to_gpu(features1)),
            Variable(cuda.to_gpu(features2)),
            Variable(cuda.to_gpu(move)),
            cuda.to_gpu(result - value)
            )

# train
itr = 0
sum_loss = 0
eval_interval = 1000
for e in range(args.epoch):
    np.random.shuffle(train_data)

    itr_epoch = 0
    sum_loss_epoch = 0
    for i in range(0, len(train_data) - args.batchsize, args.batchsize):
        x1, x2, t, z = mini_batch(train_data[i:i+args.batchsize])
        y = model(x1, x2)

        model.cleargrads()
        loss = softmax_cross_entropy_with_weight(y, t, z)
        loss.backward()
        optimizer.update()

        itr += 1
        sum_loss += loss.data
        itr_epoch += 1
        sum_loss_epoch += loss.data

        # print train loss
        if optimizer.t % eval_interval == 0:
            logging.info('epoch = {}, iteration = {}, loss = {}'.format(optimizer.epoch + 1, optimizer.t, sum_loss / itr))
            itr = 0
            sum_loss = 0

    # print train loss for each epoch
    logging.info('epoch = {}, iteration = {}, train loss avr = {}'.format(optimizer.epoch + 1, optimizer.t, sum_loss_epoch / itr_epoch))

    optimizer.new_epoch()

print('save the model')
serializers.save_npz(args.model, model)
print('save the optimizer')
serializers.save_npz(args.state, optimizer)
