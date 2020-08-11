import torch

from dlshogi import serializers
from dlshogi.policy_value_network import *
from dlshogi.common import *

import cppshogi
from cshogi import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PolicyValueNetwork()
model.to(device)

def load_model(file):
    print('Load model from', file)
    serializers.load_npz(file, model)

def predict(features1, features2):
    model.eval()
    with torch.no_grad():
        x1 = torch.tensor(features1).to(device)
        x2 = torch.tensor(features2).to(device)

        y1, y2 = model((x1, x2))

        return y1.cpu(), y2.cpu()

load_model(r"F:\model\model_rl_val_wideresnet10_selfplay_239")

#serializers.save_npz(r"R:\model", model)

board = Board()
hcpe = np.zeros(2, HuffmanCodedPosAndEval)

board.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
board.to_hcp(hcpe[0]['hcp'])

board.set_sfen("lnsgkgsnl/1r7/ppppppbpp/6pP1/9/9/PPPPPPP1P/1B5R1/LNSGKGSNL w - 1")
board.to_hcp(hcpe[1]['hcp'])

features1 = np.empty((len(hcpe), FEATURES1_NUM, 9, 9), dtype=np.float32)
features2 = np.empty((len(hcpe), FEATURES2_NUM, 9, 9), dtype=np.float32)
result = np.empty((len(hcpe)), dtype=np.float32)
cppshogi.hcpe_decode_with_result(hcpe, features1, features2, result)

y1, y2 = predict(features1, features2)

torch.set_printoptions(profile="full")
print(y1)
print(y2)
