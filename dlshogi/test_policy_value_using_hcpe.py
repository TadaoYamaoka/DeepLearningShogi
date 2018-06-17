import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from dlshogi.policy_value_network import *
from dlshogi.common import *
from dlshogi.sigmoid_cross_entropy2 import sigmoid_cross_entropy2

import cppshogi

import argparse

import logging

parser = argparse.ArgumentParser(description='Test policy network using hcpe')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--batchsize', '-b', type=int, default=640, help='Number of positions in each mini-batch')
parser.add_argument('initmodel', help='Initialize the model from given file')
parser.add_argument('--val_lambda', type=float, default=0.333, help='regularization factor')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.DEBUG)

model = PolicyValueNetwork()
model.to_gpu()

optimizer = optimizers.SGD()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)

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
            Variable(cuda.to_gpu(z)),
            Variable(cuda.to_gpu(value.reshape((len(value), 1))))
            )

# print train loss for each epoch
itr_test = 0
sum_test_loss1 = 0
sum_test_loss2 = 0
sum_test_loss3 = 0
sum_test_loss = 0
sum_test_accuracy1 = 0
sum_test_accuracy2 = 0
for i in range(0, len(test_data) - args.batchsize, args.batchsize):
    x1, x2, t1, t2, z, value = mini_batch(test_data[i:i+args.batchsize])
    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y1, y2 = model(x1, x2)
    itr_test += 1
    loss1 = F.mean(F.softmax_cross_entropy(y1, t1, reduce='no') * z)
    loss2 = F.sigmoid_cross_entropy(y2, t2)
    loss3 = sigmoid_cross_entropy2(y2, value)
    loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3
    sum_test_loss1 += loss1.data
    sum_test_loss2 += loss2.data
    sum_test_loss3 += loss3.data
    sum_test_loss += loss.data
    sum_test_accuracy1 += F.accuracy(y1, t1).data
    sum_test_accuracy2 += F.binary_accuracy(y2, t2).data
logging.info('test_loss = {}, {}, {}, {}, test accuracy = {}, {}'.format(
    sum_test_loss1 / itr_test, sum_test_loss2 / itr_test, sum_test_loss3 / itr_test, sum_test_loss / itr_test,
    sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test))
