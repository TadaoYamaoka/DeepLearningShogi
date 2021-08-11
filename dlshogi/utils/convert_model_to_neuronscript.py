import argparse
import torch
import torch_neuron

import torch.nn as nn
from dlshogi.common import *
from dlshogi.network.policy_value_network import policy_value_network
from dlshogi import serializers
from dlshogi import cppshogi

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('neuronscript')
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
model.eval()

def mini_batch(hcpevec):
    features1 = np.empty((len(hcpevec), FEATURES1_NUM, 9, 9), dtype=np.float32)
    features2 = np.empty((len(hcpevec), FEATURES2_NUM, 9, 9), dtype=np.float32)
    move = np.empty((len(hcpevec)), dtype=np.int64)
    result = np.empty((len(hcpevec)), dtype=np.float32)
    value = np.empty((len(hcpevec)), dtype=np.float32)

    cppshogi.hcpe_decode_with_value(hcpevec, features1, features2, move, result, value)

    z = result.astype(np.float32) - value + 0.5

    return (torch.tensor(features1),
            torch.tensor(features2),
            torch.tensor(move.astype(np.int64)),
            torch.tensor(result.reshape((len(hcpevec), 1))),
            torch.tensor(z),
            torch.tensor(value.reshape((len(value), 1)))
            )

batchsize = 1
hcpevec = np.array([([ 88, 164,  73,  33,  12, 215,  87,  33, 126, 142,  77,  33,  44, 175,  66, 120,  20, 194, 171,  16, 158,  77,  33,  44, 215,  95,  33,  62, 142,  73,  33,  12], 0, 7739, 1, 0)] * batchsize, HuffmanCodedPosAndEval)
x1, x2, t1, t2, z, value = mini_batch(hcpevec)

model_neuron = torch.neuron.trace(model, example_inputs=[x1, x2], dynamic_batch_size=True)

model_neuron.save(args.neuronscript)