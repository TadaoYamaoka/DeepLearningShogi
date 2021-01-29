import numpy as np
import torch
import torch.nn.functional as F

from dlshogi.common import *
from dlshogi import serializers

from dlshogi import cppshogi

import argparse
import random
import os

import cshogi

import logging

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, default='model', help='model file name')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--network', type=str, default='resnet10_swish', choices=['resnet10_swish'], help='network type')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
parser.add_argument('--index', '-i', type=int)
args = parser.parse_args()

if args.network == 'resnet10_swish':
    from dlshogi.policy_value_network_resnet10_swish_att import *
else:
    from dlshogi.policy_value_network import *

if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = PolicyValueNetwork()
model.to(device)

# Init/Resume
print('Load model from', args.model)
serializers.load_npz(args.model, model)

logging.debug('read test data')
logging.debug(args.test_data)
test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)

# mini batch
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

model.eval()
with torch.no_grad():
    if args.index is not None:
        i = args.index
    else:
        i = random.randint(0, len(test_data) - 1)
    print(f'index\t{i}')

    x1, x2, t1, t2, z, value = mini_batch(test_data[i:i+1])
    y1, y2, _, _ = model(x1, x2)

    logits = y1.cpu().numpy()
    value = torch.sigmoid(y2).cpu().numpy()
    att_p = model.att_p.cpu().numpy()
    att_v = model.att_v.cpu().numpy()

    board = cshogi.Board()
    board.set_hcp(test_data[i]['hcp'])
    print(board.sfen())

    logits_ = torch.zeros(len(board.legal_moves), dtype=torch.float32)
    for j, m in enumerate(board.legal_moves):
        label = cppshogi.move_to_label(m, board.turn)
        logits_[j] = logits[0, label].item()
    probabilities = F.softmax(logits_, dim=0)

    print('policy')
    for m, p in zip(board.legal_moves, probabilities):
        print(f"{cshogi.move_to_usi(m)}\t{p}")

    print(f"value\t{value[0]}")

    print('policy attention map')
    for rank in range(9):
        for file in reversed(range(9)):
            print(att_p[0, 0, file, rank], end='\t')
        print()

    print('value attention map')
    for rank in range(9):
        for file in reversed(range(9)):
            print(att_v[0, 0, file, rank], end='\t')
        print()