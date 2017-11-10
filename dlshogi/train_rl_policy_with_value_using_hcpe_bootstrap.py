import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from dlshogi.policy_value_network import *
from dlshogi.common import *

import cppshogi

import argparse
import random

import logging

parser = argparse.ArgumentParser(description='Traning RL policy network using hcpe')
parser.add_argument('train_data', type=str, help='train data file')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--batchsize', '-b', type=int, default=64, help='Number of positions in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--model', type=str, default='model_rl_val_hcpe', help='model file name')
parser.add_argument('--state', type=str, default='state_rl_val_hcpe', help='state file name')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weightdecay_rate', type=float, default=0.00001, help='weightdecay rate')
parser.add_argument('--val_lambda', type=float, default=0.5, help='regularization factor')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)
logging.info('batchsize={}'.format(args.batchsize))
logging.info('MomentumSGD(lr={})'.format(args.lr))
logging.info('WeightDecay(rate={})'.format(args.weightdecay_rate))

model = PolicyValueNetwork()
model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=args.lr)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(args.weightdecay_rate))

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

    z = result.astype(np.float32) - value + 1.0

    return (Variable(cuda.to_gpu(features1)),
            Variable(cuda.to_gpu(features2)),
            Variable(cuda.to_gpu(move)),
            result.reshape((len(hcpevec), 1)),
            Variable(cuda.to_gpu(z)),
            Variable(cuda.to_gpu(value.reshape((len(value), 1))))
            )

def cross_entropy(p, q):
    return F.mean(-p * F.log(q + 1.0e-16) - (1 - p) * F.log(1 - q + 1.0e-16))

# train
itr = 0
sum_loss1 = 0
sum_loss2 = 0
sum_loss3 = 0
sum_loss = 0
eval_interval = 1000
for e in range(args.epoch):
    np.random.shuffle(train_data)

    itr_epoch = 0
    sum_loss_epoch = 0
    for i in range(0, len(train_data) - args.batchsize, args.batchsize):
        x1, x2, t1, t2_np, z, value = mini_batch(train_data[i:i+args.batchsize])
        y1, y2 = model(x1, x2)
        tanh_y2 = F.tanh(y2)
        t2 = Variable(cuda.to_gpu(t2_np.astype(np.float32)))

        model.cleargrads()
        loss1 = F.mean(F.softmax_cross_entropy(y1, t1, reduce='no') * z)
        loss2 = F.mean_squared_error(tanh_y2, t2)
        loss3 = F.mean_squared_error(tanh_y2, value)
        loss = loss1 + loss2 + args.val_lambda * loss3
        loss.backward()
        optimizer.update()

        itr += 1
        sum_loss1 += loss1.data
        sum_loss2 += loss2.data
        sum_loss3 += loss3.data
        sum_loss += loss.data
        itr_epoch += 1
        sum_loss_epoch += loss.data

        # print train loss
        if optimizer.t % eval_interval == 0:
            x1, x2, t1, t2_np, z, value = mini_batch(np.random.choice(test_data, 640))
            with chainer.no_backprop_mode():
                with chainer.using_config('train', False):
                    y1, y2 = model(x1, x2)
            tanh_y2 = F.tanh(y2)
            t2 = Variable(cuda.to_gpu(t2_np.astype(np.float32)))
            loss1 = F.mean(F.softmax_cross_entropy(y1, t1, reduce='no') * z)
            loss2 = F.mean_squared_error(tanh_y2, t2)
            loss3 = F.mean_squared_error(tanh_y2, value)
            loss = loss1 + loss2 + args.val_lambda * loss3
            t2_np[t2_np < 0] = 0 # -1 to 0 for binary_accuracy
            logging.info('epoch = {}, iteration = {}, loss = {}, {}, {}, {}, test loss = {}, {}, {}, {}, test accuracy = {}, {}'.format(
                optimizer.epoch + 1, optimizer.t,
                sum_loss1 / itr, sum_loss2 / itr, sum_loss3 / itr, sum_loss / itr,
                loss1.data, loss2.data, loss3.data, loss.data,
                F.accuracy(y1, t1).data, F.binary_accuracy(y2, Variable(cuda.to_gpu(t2_np))).data))
            itr = 0
            sum_loss1 = 0
            sum_loss2 = 0
            sum_loss3 = 0
            sum_loss = 0

    # print train loss for each epoch
    itr_test = 0
    sum_test_loss1 = 0
    sum_test_loss2 = 0
    sum_test_loss3 = 0
    sum_test_loss = 0
    sum_test_accuracy1 = 0
    sum_test_accuracy2 = 0
    for i in range(0, len(test_data) - args.batchsize, args.batchsize):
        x1, x2, t1, t2_np, z, value = mini_batch(test_data[i:i+args.batchsize])
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                y1, y2 = model(x1, x2)
        tanh_y2 = F.tanh(y2)
        t2 = Variable(cuda.to_gpu(t2_np.astype(np.float32)))
        itr_test += 1
        loss1 = F.mean(F.softmax_cross_entropy(y1, t1, reduce='no') * z)
        loss2 = F.mean_squared_error(tanh_y2, t2)
        loss3 = F.mean_squared_error(tanh_y2, value)
        loss = loss1 + loss2 + args.val_lambda * loss3
        sum_test_loss1 += loss1.data
        sum_test_loss2 += loss2.data
        sum_test_loss3 += loss3.data
        sum_test_loss += loss.data
        sum_test_accuracy1 += F.accuracy(y1, t1).data
        t2_np[t2_np < 0] = 0 # -1 to 0 for binary_accuracy
        sum_test_accuracy2 += F.binary_accuracy(y2, Variable(cuda.to_gpu(t2_np))).data
    logging.info('epoch = {}, iteration = {}, train loss avr = {}, test_loss = {}, {}, {}, {}, test accuracy = {}, {}'.format(
        optimizer.epoch + 1, optimizer.t, sum_loss_epoch / itr_epoch,
        sum_test_loss1 / itr_test, sum_test_loss2 / itr_test, sum_test_loss3 / itr_test, sum_test_loss / itr_test,
        sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test))

    optimizer.new_epoch()

print('save the model')
serializers.save_npz(args.model, model)
print('save the optimizer')
serializers.save_npz(args.state, optimizer)
