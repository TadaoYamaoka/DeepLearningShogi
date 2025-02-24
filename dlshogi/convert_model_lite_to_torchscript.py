import torch
import torch.nn.functional as F

from dlshogi.common import *
from dlshogi.network.lite_value_network import LiteValueNetwork
from dlshogi import serializers
from dlshogi import cppshogi

import argparse
import sys

def main(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('torchscript')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
    parser.add_argument('--dims', type=int, nargs=3, default=(16, 4, 32), help='network dimensions')
    args = parser.parse_args(argv)

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = LiteValueNetwork(args.dims)
    model.to(device)

    serializers.load_npz(args.model, model)
    model.eval()
    
    def mini_batch(hcpevec):
        features1 = torch.empty(
            (len(hcpevec), 9 * 9), dtype=torch.long, pin_memory=True
        )
        features2 = torch.empty(
            (len(hcpevec), FEATURES2_NUM), dtype=torch.long, pin_memory=True
        )
        result = torch.empty(
            (len(hcpevec), 1), dtype=torch.float32, pin_memory=True
        )
        value = torch.empty(
            (len(hcpevec), 1), dtype=torch.float32, pin_memory=True
        )

        cppshogi.hcpe_decode_lite(
            hcpevec, features1.numpy(), features2.numpy(), result.numpy(), value.numpy()
        )

        return (
            features1.to(device),
            features2.to(device),
            result.to(device),
            value.to(device),
        )

    batchsize = 1
    hcpevec = np.array([([ 88, 164,  73,  33,  12, 215,  87,  33, 126, 142,  77,  33,  44, 175,  66, 120,  20, 194, 171,  16, 158,  77,  33,  44, 215,  95,  33,  62, 142,  73,  33,  12], 0, 7739, 1, 0)] * batchsize, HuffmanCodedPosAndEval)
    x1, x2, _, _ = mini_batch(hcpevec)

    traced_model = torch.jit.trace(model, (x1, x2))
    traced_model = torch.jit.freeze(traced_model)
    traced_model.save(args.torchscript)

    print(traced_model.graph)

if __name__ == '__main__':
    main(*sys.argv[1:])
