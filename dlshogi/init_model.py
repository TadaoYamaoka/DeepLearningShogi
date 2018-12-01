import numpy as np
import chainer
from chainer import optimizers, serializers

from dlshogi.policy_value_network import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='model', help='model file name')
parser.add_argument('--state', type=str, default='state', help='state file name')
args = parser.parse_args()

model = PolicyValueNetwork()

optimizer = optimizers.MomentumSGD()
optimizer.setup(model)

model.cleargrads()
optimizer.update()

print('save the model')
serializers.save_npz(args.model, model)
print('save the optimizer')
serializers.save_npz(args.state, optimizer)
