import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from dlshogi.policy_network import *
from dlshogi.value_network import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('policy_model', type=str)
parser.add_argument('value_model', type=str)
args = parser.parse_args()

policy_model = PolicyNetwork()
value_model = ValueNetwork()

print('Load policy model from', args.policy_model)
serializers.load_npz(args.policy_model, policy_model)

print('value model params')
value_dict = {}
for path, param in value_model.namedparams():
    print(path, param.data.shape)
    value_dict[path] = param

print('policy model params')
for path, param in policy_model.namedparams():
    print(path, param.data.shape)
    value_dict[path].data = param.data

print('save the model')
serializers.save_npz(args.value_model, value_model)
