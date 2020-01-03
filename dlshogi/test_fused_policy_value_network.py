import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import serializers
import chainer.links as L

from dlshogi.common import *

from dlshogi.policy_value_network import *
from dlshogi.fused_policy_value_network import *

import cppshogi

import argparse
import random

import logging

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('fused_model')
parser.add_argument('test_data')
parser.add_argument('--testbatchsize', type=int, default=128)
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=None, level=logging.INFO)

model = PolicyValueNetwork()
serializers.load_npz(args.model, model)
model.to_gpu()

fused_model = FusedPolicyValueNetwork()
serializers.load_npz(args.fused_model, fused_model)
fused_model.to_gpu()

test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)
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

logging.info('start')
itr_test = 0
sum_test_accuracy1 = 0
sum_test_accuracy2 = 0
for i in range(0, len(test_data) - args.testbatchsize, args.testbatchsize):
    x1, x2, t1, t2, z, value = mini_batch(test_data[i:i+args.testbatchsize])
    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y1, y2 = model(x1, x2)
    itr_test += 1
    sum_test_accuracy1 += F.accuracy(y1, t1).data
    sum_test_accuracy2 += F.binary_accuracy(y2, t2).data

logging.info('test accuracy = {}, {}'.format(
    sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test))


logging.info('start fused model')
itr_test = 0
sum_test_accuracy1 = 0
sum_test_accuracy2 = 0
for i in range(0, len(test_data) - args.testbatchsize, args.testbatchsize):
    x1, x2, t1, t2, z, value = mini_batch(test_data[i:i+args.testbatchsize])
    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y1, y2 = fused_model(x1, x2)
    itr_test += 1
    sum_test_accuracy1 += F.accuracy(y1, t1).data
    sum_test_accuracy2 += F.binary_accuracy(y2, t2).data

logging.info('test accuracy = {}, {}'.format(
    sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test))
