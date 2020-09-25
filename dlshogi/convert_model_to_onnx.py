import torch.onnx
import torch.nn.functional as F

from dlshogi.common import *
from dlshogi import serializers
from dlshogi import cppshogi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('onnx')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
parser.add_argument('--network', type=str, default='wideresnet10', choices=['wideresnet10', 'wideresnet15', 'senet10', 'resnet10_swish'])
parser.add_argument('--fixed_batchsize', type=int)
args = parser.parse_args()

if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if args.network == 'wideresnet10':
    from dlshogi.policy_value_network import *
elif args.network == 'wideresnet15':
    from dlshogi.policy_value_network_wideresnet15 import *
elif args.network == 'senet10':
    from dlshogi.policy_value_network_senet10 import *
elif args.network == 'resnet10_swish':
    from dlshogi.policy_value_network_resnet10_swish import *
baseclass = PolicyValueNetwork

class PolicyValueNetworkAddSigmoid(baseclass):
    def __init__(self):
        super(PolicyValueNetworkAddSigmoid, self).__init__()

    def __call__(self, x1, x2):
        y1, y2 = super(PolicyValueNetworkAddSigmoid, self).__call__(x1, x2)
        return y1, torch.sigmoid(y2)

model = PolicyValueNetworkAddSigmoid()
if args.network.endswith('_swish'):
    model.set_swish(False)
model.to(device)

serializers.load_npz(args.model, model)
model.eval()

def mini_batch(hcpevec):
    features1 = np.empty((len(hcpevec), FEATURES1_NUM, 9, 9), dtype=np.float32)
    features2 = np.empty((len(hcpevec), FEATURES2_NUM, 9, 9), dtype=np.float32)
    move = np.empty((len(hcpevec)), dtype=np.int32)
    result = np.empty((len(hcpevec)), dtype=np.float32)
    value = np.empty((len(hcpevec)), dtype=np.float32)

    cppshogi.hcpe_decode_with_value(hcpevec, features1, features2, move, result, value)

    z = result.astype(np.float32) - value + 0.5

    return (torch.tensor(features1).to(device),
            torch.tensor(features2).to(device),
            torch.tensor(move.astype(np.int64)).to(device),
            torch.tensor(result.reshape((len(hcpevec), 1))).to(device),
            torch.tensor(z).to(device),
            torch.tensor(value.reshape((len(value), 1))).to(device)
            )

batchsize = 1 if args.fixed_batchsize is None else args.fixed_batchsize
hcpevec = np.array([([ 88, 164,  73,  33,  12, 215,  87,  33, 126, 142,  77,  33,  44, 175,  66, 120,  20, 194, 171,  16, 158,  77,  33,  44, 215,  95,  33,  62, 142,  73,  33,  12], 0, 7739, 1, 0)] * batchsize, HuffmanCodedPosAndEval)
x1, x2, t1, t2, z, value = mini_batch(hcpevec)

if args.fixed_batchsize is None:
    torch.onnx.export(model, (x1, x2), args.onnx,
        verbose = True,
        do_constant_folding = True,
        input_names = ['input1', 'input2'],
        output_names = ['output_policy', 'output_value'],
        dynamic_axes={
            'input1' : {0 : 'batch_size'},
            'input2' : {0 : 'batch_size'},
            'output_policy' : {0 : 'batch_size'},
            'output_value' : {0 : 'batch_size'},
            })
else:
    torch.onnx.export(model, (x1, x2), args.onnx,
        verbose = True,
        do_constant_folding = True,
        input_names = ['input1', 'input2'],
        output_names = ['output_policy', 'output_value'])
