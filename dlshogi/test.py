import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from dlshogi.common import *
from dlshogi import serializers
from dlshogi.data_loader import DataLoader
from dlshogi.network.policy_value_network import policy_value_network

import argparse
import random
import os
import sys

import logging

def main(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='model', help='model file name')
    parser.add_argument('test_data', type=str, help='test data file')
    parser.add_argument('--testbatchsize', type=int, default=640, help='Number of positions in each test mini-batch')
    parser.add_argument('--network', default='resnet10_swish', help='network type')
    parser.add_argument('--log', default=None, help='log file path')
    parser.add_argument('--val_lambda', type=float, default=0.333, help='regularization factor')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = policy_value_network(args.network)
    model.to(device)

    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

    # Init/Resume
    print('Load model from', args.model)
    serializers.load_npz(args.model, model)

    logging.debug('read test data')
    logging.debug(args.test_data)
    test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)

    logging.info('test position num = {}'.format(len(test_data)))

    test_dataloader = DataLoader(test_data, args.testbatchsize, device)

    def accuracy(y, t):
        return (torch.max(y, 1)[1] == t).sum().item() / len(t)

    def binary_accuracy(y, t):
        pred = y >= 0
        truth = t >= 0.5
        return pred.eq(truth).sum().item() / len(t)

    itr_test = 0
    sum_test_loss1 = 0
    sum_test_loss2 = 0
    sum_test_loss3 = 0
    sum_test_loss = 0
    sum_test_accuracy1 = 0
    sum_test_accuracy2 = 0
    sum_test_entropy1 = 0
    sum_test_entropy2 = 0
    model.eval()
    with torch.no_grad():
        for x1, x2, t1, t2, value in test_dataloader:
            y1, y2 = model(x1, x2)

            itr_test += 1
            loss1 = cross_entropy_loss(y1, t1).mean()
            loss2 = bce_with_logits_loss(y2, t2)
            loss3 = bce_with_logits_loss(y2, value)
            loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3
            sum_test_loss1 += loss1.item()
            sum_test_loss2 += loss2.item()
            sum_test_loss3 += loss3.item()
            sum_test_loss += loss.item()
            sum_test_accuracy1 += accuracy(y1, t1)
            sum_test_accuracy2 += binary_accuracy(y2, t2)

            entropy1 = (- F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1)
            sum_test_entropy1 += entropy1.mean().item()

            p2 = y2.sigmoid()
            #entropy2 = -(p2 * F.log(p2) + (1 - p2) * F.log(1 - p2))
            log1p_ey2 = F.softplus(y2)
            entropy2 = -(p2 * (y2 - log1p_ey2) + (1 - p2) * -log1p_ey2)
            sum_test_entropy2 +=entropy2.mean().item()

        logging.info('test_loss = {:.08f}, {:.08f}, {:.08f}, {:.08f}, test accuracy = {:.08f}, {:.08f}, test entropy = {:.08f}, {:.08f}'.format(
            sum_test_loss1 / itr_test, sum_test_loss2 / itr_test, sum_test_loss3 / itr_test, sum_test_loss / itr_test,
            sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test,
            sum_test_entropy1 / itr_test, sum_test_entropy2 / itr_test))

if __name__ == '__main__':
    main(*sys.argv[1:])
