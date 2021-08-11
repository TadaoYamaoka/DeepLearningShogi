import argparse
import torch
import torch.nn as nn
from dlshogi.common import *
from dlshogi.network.policy_value_network import policy_value_network
from dlshogi import serializers
from dlshogi import cppshogi

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('torchscript')
parser.add_argument('--network', default='resnet10_swish')
args = parser.parse_args()

model = policy_value_network(args.network)
serializers.load_npz(args.model, model)

class PolicyValueNetworkAddSigmoid(nn.Module):
    def __init__(self, model):
        super(PolicyValueNetworkAddSigmoid, self).__init__()
        self.model = model
        
    def forward(self, x1, x2):
        y1, y2 = self.model(x1, x2)
        return y1, torch.sigmoid(y2)

model = PolicyValueNetworkAddSigmoid(model)

torchscript = torch.jit.script(model)

torchscript.save(args.torchscript)
