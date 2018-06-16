import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import serializers
import chainer.functions as F

from dlshogi.policy_value_network import *
from dlshogi.common import *

import cppshogi

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_data', type=str, default=r'H:\src\DeepLearningShogi\x64\Release_NoOpt\test.hcpe', help='test data file')
parser.add_argument('--initmodel', '-m', default=r'H:\src\DeepLearningShogi\dlshogi\model_rl_val_wideresnet10_110_1', help='Initialize the model from given file')
args = parser.parse_args()

model = PolicyValueNetwork()
model.to_gpu()

print('Load model from', args.initmodel)
serializers.load_npz(args.initmodel, model)

hcpevec = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)

features1 = np.empty((len(hcpevec), FEATURES1_NUM, 9, 9), dtype=np.float32)
features2 = np.empty((len(hcpevec), FEATURES2_NUM, 9, 9), dtype=np.float32)
move = np.empty((len(hcpevec)), dtype=np.int32)
result = np.empty((len(hcpevec)), dtype=np.int32)
value = np.empty((len(hcpevec)), dtype=np.float32)

cppshogi.hcpe_decode_with_value(hcpevec, features1, features2, move, result, value)

x1 = Variable(cuda.to_gpu(features1))
x2 = Variable(cuda.to_gpu(features2))

with chainer.no_backprop_mode():
    with chainer.using_config('train', False):
        y1, y2 = model(x1, x2)

print(y1.data)
print(F.sigmoid(y2).data)