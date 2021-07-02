import torch.onnx
import torch.nn.functional as F

from dlshogi.common import *
from dlshogi.network.policy_value_network import policy_value_network
from dlshogi import serializers
from dlshogi import cppshogi

import argparse
import sys

def main(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('onnx')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
    parser.add_argument('--network', default='resnet10_swish')
    parser.add_argument('--fixed_batchsize', type=int)
    parser.add_argument('--remove_aux', action='store_true')
    args = parser.parse_args(argv)

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = policy_value_network(args.network, add_sigmoid=True)
    if args.network[-6:] == '_swish':
        model.set_swish(False)
    model.to(device)

    serializers.load_npz(args.model, model, args.remove_aux)
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

if __name__ == '__main__':
    main(*sys.argv[1:])
