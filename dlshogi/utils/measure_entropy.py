from chainer import serializers
import chainer.functions as F

import numpy as np

from dlshogi.common import *
from dlshogi.policy_value_network import *
from dlshogi import cppshogi

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('hcpe')
args = parser.parse_args()

model = PolicyValueNetwork()
model.to_gpu()

serializers.load_npz(args.model, model)

hcpes = np.fromfile(args.hcpe, HuffmanCodedPosAndEval)

BATCH_SIZE = 256

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
            )

sum_entropy1 = 0.0
sum_entropy2 = 0.0
n = 0
for i in range(0, len(hcpes) - BATCH_SIZE, BATCH_SIZE):
    x1, x2 = mini_batch(hcpes[i:i+BATCH_SIZE])
    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y1, y2 = model(x1, x2)

        p1 = F.softmax(y1)
        #entropy1 = F.sum(- p1 * F.log(p1), axis=1)
        y1_max = F.max(y1, axis=1, keepdims=True)
        log_p1 = y1 - (F.log(F.sum(F.exp(y1 - y1_max), axis=1, keepdims=True)) + y1_max)
        entropy1 = F.sum(- p1 * log_p1, axis=1)
        sum_entropy1 += F.mean(entropy1).data

        p2 = F.sigmoid(y2)
        #entropy2 = -(p2 * F.log(p2) + (1 - p2) * F.log(1 - p2))
        log1p_ey2 = F.log1p(F.exp(y2))
        entropy2 = -(p2 * (y2 - log1p_ey2) + (1 - p2) * -log1p_ey2)
        sum_entropy2 += F.mean(entropy2).data

    n += 1

avr_entropy1 = sum_entropy1 / n
avr_entropy2 = sum_entropy2 / n
print(avr_entropy1, avr_entropy2)
