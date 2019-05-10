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

sum_entropy = 0.0
n = 0
for i in range(0, len(hcpes) - BATCH_SIZE, BATCH_SIZE):
    x1, x2 = mini_batch(hcpes[i:i+BATCH_SIZE])
    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y1, y2 = model(x1, x2)

    p = F.softmax(y1)
    entropy = F.sum(- p * F.log(F.clip(p, 1e-32, 1.0)), axis=1)

    sum_entropy += F.mean(entropy).data
    n += 1

avr_entropy = sum_entropy / n
print(avr_entropy)