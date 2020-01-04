import numpy as np
import torch

from dlshogi.common import *

from dlshogi.policy_value_network import *
from dlshogi.fused_policy_value_network import *
from dlshogi import serializers

import cppshogi

import argparse
import random

import logging

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('fused_model')
parser.add_argument('test_data')
parser.add_argument('--testbatchsize', type=int, default=256)
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=None, level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PolicyValueNetwork()
serializers.load_npz(args.model, model)
model.to(device)

fused_model = FusedPolicyValueNetwork()
serializers.load_npz(args.fused_model, fused_model)
fused_model.to(device)

test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)
logging.info('test position num = {}'.format(len(test_data)))

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

def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)

def binary_accuracy(y, t):
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)

model.eval()
fused_model.eval()
with torch.no_grad():
    logging.info('start')
    itr_test = 0
    sum_test_accuracy1 = 0
    sum_test_accuracy2 = 0
    for i in range(0, len(test_data) - args.testbatchsize, args.testbatchsize):
        x1, x2, t1, t2, z, value = mini_batch(test_data[i:i+args.testbatchsize])
        y1, y2 = model(x1, x2)
        itr_test += 1
        sum_test_accuracy1 += accuracy(y1, t1)
        sum_test_accuracy2 += binary_accuracy(y2, t2)

    logging.info('test accuracy = {}, {}'.format(
        sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test))


    logging.info('start fused model')
    itr_test = 0
    sum_test_accuracy1 = 0
    sum_test_accuracy2 = 0
    for i in range(0, len(test_data) - args.testbatchsize, args.testbatchsize):
        x1, x2, t1, t2, z, value = mini_batch(test_data[i:i+args.testbatchsize])
        y1, y2 = fused_model(x1, x2)
        itr_test += 1
        sum_test_accuracy1 += accuracy(y1, t1)
        sum_test_accuracy2 += binary_accuracy(y2, t2)

    logging.info('test accuracy = {}, {}'.format(
        sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test))
