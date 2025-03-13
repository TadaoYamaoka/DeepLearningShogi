import torch
import torch.nn.functional as F
from torch.quantization import fuse_modules

from dlshogi.common import *
from dlshogi.network.lite_value_network import LiteValueNetwork
from dlshogi import serializers
from dlshogi import cppshogi

import argparse
import sys
import struct

def main(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('bin')
    parser.add_argument('--dims', type=int, nargs=3, default=(16, 4, 32), help='network dimensions')
    args = parser.parse_args(argv)

    model = LiteValueNetwork(args.dims)

    serializers.load_npz(args.model, model)
    model.eval()

    fuse_modules(model, [['l2_1', 'bn2_1'], ['l2_2', 'bn2_2']], inplace=True)
    
    state_dict = model.state_dict()
    
    with open(args.bin, "wb") as f:
        # 全パラメータ数をunsigned intとして書き出し
        num_params = len(state_dict)
        f.write(struct.pack("I", num_params))
        
        for key, tensor in state_dict.items():
            print(key)

            # CPU に移動、float32 に変換
            tensor = tensor.detach().cpu().float()
            
            # パラメータ名を UTF-8 でエンコードし、バイト長とともに書き出す
            key_bytes = key.encode('utf-8')
            f.write(struct.pack("I", len(key_bytes)))
            f.write(key_bytes)
            
            # テンソルの形状情報を書き出す
            shape = tensor.shape
            f.write(struct.pack("I", len(shape)))  # 次元数
            for dim in shape:
                f.write(struct.pack("I", dim))
                
            # 要素数を書き出す
            numel = tensor.numel()
            f.write(struct.pack("I", numel))
            
            # テンソルデータを書き出す（float32 の連続バイト列）
            f.write(tensor.numpy().tobytes())

if __name__ == '__main__':
    main(*sys.argv[1:])
