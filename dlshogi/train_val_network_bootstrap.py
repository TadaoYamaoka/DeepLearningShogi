import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from dlshogi.value_network import *
from dlshogi.common import *

import cppshogi

import argparse
import random
import os

import logging

parser = argparse.ArgumentParser(description='Train value network')
parser.add_argument('train_data', type=str, help='train data file')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--batchsize', '-b', type=int, default=256, help='Number of positions in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--model', type=str, default='model_val', help='model file name')
parser.add_argument('--state', type=str, default='state_val', help='state file name')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--val_lambda', type=float, default=0.5, help='regularization factor')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

model = ValueNetwork()
model.to_gpu()

optimizer = optimizers.SGD(lr=args.lr)
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
test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)
logging.debug('read teacher data end')

logging.info('train position num = {}'.format(len(train_data)))
logging.info('test position num = {}'.format(len(test_data)))

# mini batch
def mini_batch(hcpevec):
    features1 = np.empty((len(hcpevec), FEATURES1_NUM, 9, 9), dtype=np.float32)
    features2 = np.empty((len(hcpevec), FEATURES2_NUM, 9, 9), dtype=np.float32)
    move = np.empty((len(hcpevec)), dtype=np.int32)
    result = np.empty((len(hcpevec)), dtype=np.int32)
    value = np.empty((len(hcpevec)), dtype=np.float32)

    cppshogi.hcpe_decode_with_value(hcpevec, features1, features2, move, result, value)

    return (Variable(cuda.to_gpu(features1)),
            Variable(cuda.to_gpu(features2)),
            Variable(cuda.to_gpu(result.reshape((len(hcpevec), 1)))),
            Variable(cuda.to_gpu(value.reshape((len(value), 1))))
            )

def cross_entropy(p, q):
    return F.mean(-p * F.log(q) - (1 - p) * F.log(1 - q))

# train
itr = 0
sum_loss1 = 0
sum_loss2 = 0
sum_loss = 0
eval_interval = 1000
for e in range(args.epoch):
    np.random.shuffle(train_data)

    itr_epoch = 0
    sum_loss_epoch = 0
    for i in range(0, len(train_data) - args.batchsize, args.batchsize):
        x1, x2, t, value = mini_batch(train_data[i:i+args.batchsize])
        y = model(x1, x2)

        model.cleargrads()
        loss1 = F.sigmoid_cross_entropy(y, t)
        loss2 = cross_entropy(F.sigmoid(y), value)
        loss = loss1 + args.val_lambda * loss2
        loss.backward()
        optimizer.update()

        itr += 1
        sum_loss1 += loss1.data
        sum_loss2 += loss2.data
        sum_loss += loss.data
        itr_epoch += 1
        sum_loss_epoch += loss.data

        # print train loss and test accuracy
        if optimizer.t % eval_interval == 0:
            x1, x2, t, value = mini_batch(np.random.choice(test_data, 640))
            with chainer.no_backprop_mode():
                with chainer.using_config('train', False):
                    y = model(x1, x2)
            loss1 = F.sigmoid_cross_entropy(y, t)
            loss2 = cross_entropy(F.sigmoid(y), value)
            loss = loss1 + args.val_lambda * loss2
            logging.info('epoch = {}, iteration = {}, train loss = {}, {}, {}, test loss = {}, {}, {}, test accuracy = {}'.format(
                optimizer.epoch + 1, optimizer.t,
                sum_loss1 / itr, sum_loss2 / itr, sum_loss / itr,
                loss1.data, loss2.data, loss.data,
                F.binary_accuracy(y, t).data))
            itr = 0
            sum_loss1 = 0
            sum_loss2 = 0
            sum_loss = 0

    # validate test data
    itr_test = 0
    sum_test_loss1 = 0
    sum_test_loss2 = 0
    sum_test_loss = 0
    sum_test_accuracy = 0
    for i in range(0, len(test_data) - args.batchsize, args.batchsize):
        x1, x2, t, value = mini_batch(test_data[i:i+args.batchsize])
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                y = model(x1, x2)
        loss1 = F.sigmoid_cross_entropy(y, t)
        loss2 = cross_entropy(F.sigmoid(y), value)
        loss = loss1 + args.val_lambda * loss2
        itr_test += 1
        sum_test_loss1 += loss1.data
        sum_test_loss2 += loss2.data
        sum_test_loss += loss.data
        sum_test_accuracy += F.binary_accuracy(y, t).data
    logging.info('epoch = {}, iteration = {}, train loss avr = {}, test loss = {}, {}, {}, test accuracy = {}'.format(
        optimizer.epoch + 1, optimizer.t, sum_loss_epoch / itr_epoch,
        sum_test_loss1 / itr_test, sum_test_loss2 / itr_test, sum_test_loss / itr_test,
        sum_test_accuracy / itr_test))
    
    optimizer.new_epoch()

print('save the model')
serializers.save_npz(args.model, model)
print('save the optimizer')
serializers.save_npz(args.state, optimizer)
