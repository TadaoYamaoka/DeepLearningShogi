import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from dlshogi.common import *
from dlshogi.sigmoid_cross_entropy2 import sigmoid_cross_entropy2
from dlshogi.binary_accuracy2 import binary_accuracy2

import cppshogi

import argparse
import random

import logging

chainer.global_config.autotune = True

parser = argparse.ArgumentParser(description='Traning RL policy network using hcpe')
parser.add_argument('train_data', type=str, help='train data file')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--batchsize', '-b', type=int, default=1024, help='Number of positions in each mini-batch')
parser.add_argument('--testbatchsize', type=int, default=640, help='Number of positions in each test mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--network', type=str, default='wideresnet10', choices=['wideresnet10', 'wideresnet15', 'senet10'], help='network type')
parser.add_argument('--model', type=str, default='model_rl_val_hcpe', help='model file name')
parser.add_argument('--state', type=str, default='state_rl_val_hcpe', help='state file name')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weightdecay_rate', type=float, default=0.0001, help='weightdecay rate')
parser.add_argument('--beta', type=float, default=0.001, help='entropy regularization coeff')
parser.add_argument('--val_lambda', type=float, default=0.333, help='regularization factor')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
args = parser.parse_args()

if args.network == 'wideresnet15':
    from dlshogi.policy_value_network_wideresnet15 import *
if args.network == 'senet10':
    from dlshogi.policy_value_network_senet10 import *
else:
    from dlshogi.policy_value_network import *

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)
logging.info('batchsize={}'.format(args.batchsize))
logging.info('MomentumSGD(lr={})'.format(args.lr))
logging.info('WeightDecay(rate={})'.format(args.weightdecay_rate))
logging.info('entropy regularization coeff={}'.format(args.beta))
logging.info('val_lambda={}'.format(args.val_lambda))

cuda.get_device(args.gpu).use()

model = PolicyValueNetwork()
model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=args.lr)
optimizer.setup(model)
if args.weightdecay_rate > 0:
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(args.weightdecay_rate))

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
    result = np.empty((len(hcpevec)), dtype=np.float32)
    value = np.empty((len(hcpevec)), dtype=np.float32)

    cppshogi.hcpe_decode_with_value(hcpevec, features1, features2, move, result, value)

    z = result.astype(np.float32) - value + 0.5

    return (Variable(cuda.to_gpu(features1)),
            Variable(cuda.to_gpu(features2)),
            Variable(cuda.to_gpu(move)),
            Variable(cuda.to_gpu(result.reshape((len(hcpevec), 1)))),
            Variable(cuda.to_gpu(z)),
            Variable(cuda.to_gpu(value.reshape((len(value), 1))))
            )

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
    sum_loss1_epoch = 0
    sum_loss2_epoch = 0
    sum_loss3_epoch = 0
    sum_loss_epoch = 0
    for i in range(0, len(train_data) - args.batchsize, args.batchsize):
        x1, x2, t1, t2, z, value = mini_batch(train_data[i:i+args.batchsize])
        y1, y2 = model(x1, x2)

        model.cleargrads()
        loss1 = F.mean(F.softmax_cross_entropy(y1, t1, reduce='no') * z)
        if args.beta > 0:
            loss1 += args.beta * F.mean(F.sum(F.softmax(y1) * y1, axis=1))
        loss2 = sigmoid_cross_entropy2(y2, t2)
        loss3 = sigmoid_cross_entropy2(y2, value)
        loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3
        loss.backward()
        optimizer.update()

        itr += 1
        sum_loss1 += loss1.data
        sum_loss2 += loss2.data
        sum_loss3 += loss3.data
        sum_loss += loss.data
        itr_epoch += 1
        sum_loss1_epoch += loss1.data
        sum_loss2_epoch += loss2.data
        sum_loss3_epoch += loss3.data
        sum_loss_epoch += loss.data

        # print train loss
        if optimizer.t % eval_interval == 0:
            x1, x2, t1, t2, z, value = mini_batch(np.random.choice(test_data, args.testbatchsize))
            with chainer.no_backprop_mode():
                with chainer.using_config('train', False):
                    y1, y2 = model(x1, x2)

                loss1 = F.mean(F.softmax_cross_entropy(y1, t1, reduce='no') * z)
                loss2 = sigmoid_cross_entropy2(y2, t2)
                loss3 = sigmoid_cross_entropy2(y2, value)
                loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3

                logging.info('epoch = {}, iteration = {}, loss = {}, {}, {}, {}, test loss = {}, {}, {}, {}, test accuracy = {}, {}'.format(
                    optimizer.epoch + 1, optimizer.t,
                    sum_loss1 / itr, sum_loss2 / itr, sum_loss3 / itr, sum_loss / itr,
                    loss1.data, loss2.data, loss3.data, loss.data,
                    F.accuracy(y1, t1).data, binary_accuracy2(y2, t2).data))
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
    sum_test_entropy1 = 0
    sum_test_entropy2 = 0
    for i in range(0, len(test_data) - args.testbatchsize, args.testbatchsize):
        x1, x2, t1, t2, z, value = mini_batch(test_data[i:i+args.testbatchsize])
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                y1, y2 = model(x1, x2)

            itr_test += 1
            loss1 = F.mean(F.softmax_cross_entropy(y1, t1, reduce='no') * z)
            loss2 = sigmoid_cross_entropy2(y2, t2)
            loss3 = sigmoid_cross_entropy2(y2, value)
            loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3
            sum_test_loss1 += loss1.data
            sum_test_loss2 += loss2.data
            sum_test_loss3 += loss3.data
            sum_test_loss += loss.data
            sum_test_accuracy1 += F.accuracy(y1, t1).data
            sum_test_accuracy2 += binary_accuracy2(y2, t2).data

            p1 = F.softmax(y1)
            #entropy1 = F.sum(- p1 * F.log(p1), axis=1)
            y1_max = F.max(y1, axis=1, keepdims=True)
            log_p1 = y1 - (F.log(F.sum(F.exp(y1 - y1_max), axis=1, keepdims=True)) + y1_max)
            entropy1 = F.sum(- p1 * log_p1, axis=1)
            sum_test_entropy1 += F.mean(entropy1).data

            p2 = F.sigmoid(y2)
            #entropy2 = -(p2 * F.log(p2) + (1 - p2) * F.log(1 - p2))
            log1p_ey2 = F.softplus(y2)
            entropy2 = -(p2 * (y2 - log1p_ey2) + (1 - p2) * -log1p_ey2)
            sum_test_entropy2 += F.mean(entropy2).data

    logging.info('epoch = {}, iteration = {}, train loss avr = {}, {}, {}, {}, test_loss = {}, {}, {}, {}, test accuracy = {}, {}, test entropy = {}, {}'.format(
        optimizer.epoch + 1, optimizer.t,
        sum_loss1_epoch / itr_epoch, sum_loss2_epoch / itr_epoch, sum_loss3_epoch / itr_epoch, sum_loss_epoch / itr_epoch,
        sum_test_loss1 / itr_test, sum_test_loss2 / itr_test, sum_test_loss3 / itr_test, sum_test_loss / itr_test,
        sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test,
        sum_test_entropy1 / itr_test, sum_test_entropy2 / itr_test))

    optimizer.new_epoch()

print('save the model')
serializers.save_npz(args.model, model)
print('save the optimizer')
serializers.save_npz(args.state, optimizer)
