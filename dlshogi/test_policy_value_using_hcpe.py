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

parser = argparse.ArgumentParser(description='Test RL policy network using hcpe')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of positions in each mini-batch')
parser.add_argument('initmodel', help='Initialize the model from given file')
parser.add_argument('resume', help='Resume the optimization from snapshot')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

model = PolicyValueNetwork()
model.to_gpu()

optimizer = optimizers.SGD()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

logging.debug('read teacher data start')
test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)
logging.debug('read teacher data end')

logging.info('test position num = {}'.format(len(test_data)))

# mini batch
def mini_batch(hcpevec):
    features1 = np.empty((len(hcpevec), FEATURES1_NUM, 9, 9), dtype=np.float32)
    features2 = np.empty((len(hcpevec), FEATURES2_NUM, 9, 9), dtype=np.float32)
    move = np.empty((len(hcpevec)), dtype=np.int32)
    result = np.empty((len(hcpevec)), dtype=np.int32)
    value = np.empty((len(hcpevec)), dtype=np.float32)

    cppshogi.hcpe_decode_with_value(hcpevec, features1, features2, move, result, value)

    z = result.astype(np.float32) - value + 0.5

    return (Variable(cuda.to_gpu(features1)),
            Variable(cuda.to_gpu(features2)),
            Variable(cuda.to_gpu(move)),
            Variable(cuda.to_gpu(result.reshape((len(hcpevec), 1)))),
            Variable(cuda.to_gpu(z))
            )

# print train loss for each epoch
itr_test = 0
sum_test_loss = 0
sum_test_accuracy1 = 0
sum_test_accuracy2 = 0
for i in range(0, len(test_data) - args.batchsize, args.batchsize):
    x1, x2, t1, t2, z = mini_batch(test_data[i:i+args.batchsize])
    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y1, y2 = model(x1, x2)
    itr_test += 1
    sum_test_loss += loss1 + loss2
    sum_test_accuracy1 += F.accuracy(y1, t1).data
    sum_test_accuracy2 += F.binary_accuracy(y2, t2).data
logging.info('epoch = {}, iteration = {}, test accuracy1 = {}, test accuracy2 = {}'.format(optimizer.epoch + 1, optimizer.t, sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test))
