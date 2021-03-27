﻿import numpy as np
import torch
import torch.optim as optim

from dlshogi.common import *
from dlshogi import serializers
from dlshogi.swa import SWA

from dlshogi import cppshogi

import argparse
import random
import os
from concurrent.futures import ThreadPoolExecutor

import logging

parser = argparse.ArgumentParser(description='Traning RL policy network using hcpe')
parser.add_argument('train_data', type=str, nargs='+', help='train data file')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--batchsize', '-b', type=int, default=1024, help='Number of positions in each mini-batch')
parser.add_argument('--testbatchsize', type=int, default=640, help='Number of positions in each test mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--network', type=str, default='wideresnet10', choices=['wideresnet10', 'wideresnet15', 'senet10', 'resnet10_swish', 'resnet20_swish'], help='network type')
parser.add_argument('--model', type=str, default='model_rl_val_hcpe', help='model file name')
parser.add_argument('--state', type=str, default='state_rl_val_hcpe', help='state file name')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weightdecay_rate', type=float, default=0.0001, help='weightdecay rate')
parser.add_argument('--clip_grad_max_norm', type=float, default=10.0, help='max norm of the gradients')
parser.add_argument('--beta', type=float, default=0.001, help='entropy regularization coeff')
parser.add_argument('--val_lambda', type=float, default=0.333, help='regularization factor')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
parser.add_argument('--eval_interval', type=int, default=1000, help='evaluation interval')
parser.add_argument('--swa_freq', type=int, default=250)
parser.add_argument('--swa_n_avr', type=int, default=10)
parser.add_argument('--swa_lr', type=float)
parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
args = parser.parse_args()

if args.network == 'wideresnet15':
    from dlshogi.policy_value_network_wideresnet15 import *
elif args.network == 'senet10':
    from dlshogi.policy_value_network_senet10 import *
elif args.network == 'resnet10_swish':
    from dlshogi.policy_value_network_resnet10_swish import *
elif args.network == 'resnet20_swish':
    from dlshogi.policy_value_network_resnet20_swish import *
else:
    from dlshogi.policy_value_network import *

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)
logging.info('batchsize={}'.format(args.batchsize))
logging.info('MomentumSGD(lr={})'.format(args.lr))
logging.info('WeightDecay(rate={})'.format(args.weightdecay_rate))
logging.info('entropy regularization coeff={}'.format(args.beta))
logging.info('val_lambda={}'.format(args.val_lambda))

if args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")

model = PolicyValueNetwork()
model.to(device)

base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightdecay_rate, nesterov=True)
optimizer = SWA(base_optimizer, swa_start=args.swa_freq, swa_freq=args.swa_freq, swa_lr=args.swa_lr, swa_n_avr=args.swa_n_avr)
cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
if args.use_amp:
    logging.info('use amp')
    scaler = torch.cuda.amp.GradScaler()

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    checkpoint = torch.load(args.resume, map_location=device)
    epoch = checkpoint['epoch']
    t = checkpoint['t']
    base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if args.use_amp and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
else:
    epoch = 0
    t = 0

logging.debug('read teacher data')
def load_teacher(files):
    data = []
    for path in files:
        if os.path.exists(path):
            logging.debug(path)
            data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEval))
        else:
            logging.debug('{} not found, skipping'.format(path))
    return np.concatenate(data)
train_data = load_teacher(args.train_data)
logging.debug('read test data')
logging.debug(args.test_data)
test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)

logging.info('train position num = {}'.format(len(train_data)))
logging.info('test position num = {}'.format(len(test_data)))

class DataLoader:
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.torch_features1 = torch.empty((batch_size, FEATURES1_NUM, 9, 9), dtype=torch.float32, pin_memory=True)
        self.torch_features2 = torch.empty((batch_size, FEATURES2_NUM, 9, 9), dtype=torch.float32, pin_memory=True)
        self.torch_move = torch.empty((batch_size), dtype=torch.int64, pin_memory=True)
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)
        self.torch_value = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)

        self.features1 = self.torch_features1.numpy()
        self.features2 = self.torch_features2.numpy()
        self.move = self.torch_move.numpy()
        self.result = self.torch_result.numpy().reshape(-1)
        self.value = self.torch_value.numpy().reshape(-1)

        self.i = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

    def mini_batch(self, hcpevec):
        cppshogi.hcpe_decode_with_value(hcpevec, self.features1, self.features2, self.move, self.result, self.value)

        z = self.result - self.value + 0.5

        return (self.torch_features1.to(device),
                self.torch_features2.to(device),
                self.torch_move.to(device),
                self.torch_result.to(device),
                torch.tensor(z).to(device),
                self.torch_value.to(device)
                )

    def sample(self):
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))

    def pre_fetch(self):
        hcpevec = self.data[self.i:self.i+self.batch_size]
        if len(hcpevec) == 0:
            return
        self.i += self.batch_size

        self.f = self.executor.submit(self.mini_batch, hcpevec)

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        self.pre_fetch()
        return self

    def __next__(self):
        if self.i >= len(self.data) - self.batch_size + 1:
            raise StopIteration()

        result = self.f.result()
        self.pre_fetch()

        return result


train_dataloader = DataLoader(train_data, args.batchsize, shuffle=True)
test_dataloader = DataLoader(test_data, args.testbatchsize)

# for SWA bn_update
def hcpe_loader(data, batchsize):
    for x1, x2, t1, t2, z, value in DataLoader(data, batchsize):
        yield x1, x2

def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)

def binary_accuracy(y, t):
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)

# train
itr = 0
sum_loss1 = 0
sum_loss2 = 0
sum_loss3 = 0
sum_loss = 0
eval_interval = args.eval_interval
for e in range(args.epoch):
    itr_epoch = 0
    sum_loss1_epoch = 0
    sum_loss2_epoch = 0
    sum_loss3_epoch = 0
    sum_loss_epoch = 0
    for x1, x2, t1, t2, z, value in train_dataloader:
        if args.use_amp:
            amp_context = torch.cuda.amp.autocast()
            amp_context.__enter__()

        model.train()

        y1, y2 = model(x1, x2)

        model.zero_grad()
        loss1 = (cross_entropy_loss(y1, t1) * z).mean()
        if args.beta > 0:
            loss1 += args.beta * (F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1).mean()
        loss2 = bce_with_logits_loss(y2, t2)
        loss3 = bce_with_logits_loss(y2, value)
        loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3

        if args.use_amp:
            amp_context.__exit__()
            scaler.scale(loss).backward()
            if args.clip_grad_max_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_max_norm)
            optimizer.step()

        t += 1
        itr += 1
        sum_loss1 += loss1.item()
        sum_loss2 += loss2.item()
        sum_loss3 += loss3.item()
        sum_loss += loss.item()
        itr_epoch += 1
        sum_loss1_epoch += loss1.item()
        sum_loss2_epoch += loss2.item()
        sum_loss3_epoch += loss3.item()
        sum_loss_epoch += loss.item()

        # print train loss
        if t % eval_interval == 0:
            model.eval()

            x1, x2, t1, t2, z, value = test_dataloader.sample()
            with torch.no_grad():
                y1, y2 = model(x1, x2)

                loss1 = (cross_entropy_loss(y1, t1) * z).mean()
                loss2 = bce_with_logits_loss(y2, t2)
                loss3 = bce_with_logits_loss(y2, value)
                loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3

                logging.info('epoch = {}, iteration = {}, loss = {:.08f}, {:.08f}, {:.08f}, {:.08f}, test loss = {:.08f}, {:.08f}, {:.08f}, {:.08f}, test accuracy = {:.08f}, {:.08f}'.format(
                    epoch + 1, t,
                    sum_loss1 / itr, sum_loss2 / itr, sum_loss3 / itr, sum_loss / itr,
                    loss1.item(), loss2.item(), loss3.item(), loss.item(),
                    accuracy(y1, t1), binary_accuracy(y2, t2)))
            itr = 0
            sum_loss1 = 0
            sum_loss2 = 0
            sum_loss3 = 0
            sum_loss = 0

    optimizer.swap_swa_sgd()

    if args.use_amp:
        amp_context = torch.cuda.amp.autocast()
        amp_context.__enter__()

    optimizer.bn_update(hcpe_loader(train_data, args.batchsize), model)

    if args.use_amp:
        amp_context.__exit__()

    # print train loss for each epoch
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
        for x1, x2, t1, t2, z, value in test_dataloader:
            y1, y2 = model(x1, x2)

            itr_test += 1
            loss1 = (cross_entropy_loss(y1, t1) * z).mean()
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

        logging.info('epoch = {}, iteration = {}, train loss avr = {:.08f}, {:.08f}, {:.08f}, {:.08f}, test_loss = {:.08f}, {:.08f}, {:.08f}, {:.08f}, test accuracy = {:.08f}, {:.08f}, test entropy = {:.08f}, {:.08f}'.format(
            epoch + 1, t,
            sum_loss1_epoch / itr_epoch, sum_loss2_epoch / itr_epoch, sum_loss3_epoch / itr_epoch, sum_loss_epoch / itr_epoch,
            sum_test_loss1 / itr_test, sum_test_loss2 / itr_test, sum_test_loss3 / itr_test, sum_test_loss / itr_test,
            sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test,
            sum_test_entropy1 / itr_test, sum_test_entropy2 / itr_test))

    epoch += 1

    if e != args.epoch - 1:
        optimizer.swap_swa_sgd()

print('save the model')
serializers.save_npz(args.model, model)
print('save the optimizer')
state = {
    'epoch': epoch,
    't': t,
    'optimizer_state_dict': base_optimizer.state_dict(),
    }
if args.use_amp:
    state['scaler_state_dict'] = scaler.state_dict()
torch.save(state, args.state)
