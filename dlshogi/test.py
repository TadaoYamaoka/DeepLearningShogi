from chainer import cuda, Variable, serializers

from dlshogi.policy_value_network_senet10 import *
from dlshogi.common import *

import cppshogi
from cshogi import *
import numpy as np

model = PolicyValueNetwork()
model.to_gpu()

def load_model(file):
    print('Load model from', file)
    serializers.load_npz(file, model)

def predict(features1, features2):
    x1 = Variable(cuda.to_gpu(features1))
    x2 = Variable(cuda.to_gpu(features2))

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y1, y2 = model(x1, x2)

    return cuda.to_cpu(y1.data), cuda.to_cpu(y2.data)

load_model(r"F:\model\model_rl_val_senet10_50")

board = Board()
hcpe = np.zeros(2, HuffmanCodedPosAndEval)

board.set_sfen(b"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
board.to_hcp(hcpe[0]['hcp'])

board.set_sfen(b"lnsgkgsnl/1r7/ppppppbpp/6pP1/9/9/PPPPPPP1P/1B5R1/LNSGKGSNL w - 1")
board.to_hcp(hcpe[1]['hcp'])

features1 = np.empty((len(hcpe), FEATURES1_NUM, 9, 9), dtype=np.float32)
features2 = np.empty((len(hcpe), FEATURES2_NUM, 9, 9), dtype=np.float32)
result = np.empty((len(hcpe)), dtype=np.float32)
cppshogi.hcpe_decode_with_result(hcpe, features1, features2, result)

y1, y2 = predict(features1, features2)

print(y1)
