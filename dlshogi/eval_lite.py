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
    parser.add_argument('--dims', type=int, nargs=3, default=(16, 4, 32), help='network dimensions')
    args = parser.parse_args(argv)

    model = LiteValueNetwork(args.dims)

    serializers.load_npz(args.model, model)
    model.eval()
    
    def mini_batch(hcpevec):
        features1 = torch.empty(
            (len(hcpevec), 9 * 9), dtype=torch.long
        )
        features2 = torch.empty(
            (len(hcpevec), FEATURES2_NUM), dtype=torch.long
        )
        result = torch.empty(
            (len(hcpevec), 1), dtype=torch.float32
        )
        value = torch.empty(
            (len(hcpevec), 1), dtype=torch.float32
        )

        cppshogi.hcpe_decode_lite(
            hcpevec, features1.numpy(), features2.numpy(), result.numpy(), value.numpy()
        )

        return (
            features1,
            features2,
            result,
            value,
        )

    hcpevec = np.array([
        ([ 88, 164,  73,  33,  12, 215,  87,  33, 126, 142,  77,  33,  44, 175,  66, 120,  20, 194, 171,  16, 158,  77,  33,  44, 215,  95,  33,  62, 142,  73,  33,  12], 0, 7739, 1, 0),
        ([ 87, 166,  73,  33, 124, 159,  10, 128, 248,   1, 175,  18,  76,  46, 130, 141, 143,  37,  64, 192, 226,  15, 241,  56,  38, 133,  48,   0,   0,  70,  57, 188], -1258, 5042, 1, 0),
        ([ 86, 156,   9,  37,  12,  62,  74, 128,  77, 192, 143, 173,  28, 247, 227,  31,  21, 120,  55, 175,   0,   5,  48, 160, 224,  99,   0, 140,  67, 146, 228, 172], 1751, 4385, 1, 0),
        ], HuffmanCodedPosAndEval)
    x1, x2, _, _ = mini_batch(hcpevec)

    y = model(x1, x2)

    print(y.detach().cpu().numpy())

if __name__ == '__main__':
    main(*sys.argv[1:])
