import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from dlshogi.common import *
from dlshogi import serializers
from dlshogi import cppshogi
from dlshogi.tree_policy import TreePolicy

import argparse
import random
import os

import logging

parser = argparse.ArgumentParser()
parser.add_argument('train_data', type=str, nargs='+', help='train data file')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--batchsize', '-b', type=int, default=1024, help='Number of positions in each mini-batch')
parser.add_argument('--testbatchsize', type=int, default=640, help='Number of positions in each test mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--model', type=str, default='model_rl_val_hcpe', help='model file name')
parser.add_argument('--state', type=str, default='state_rl_val_hcpe', help='state file name')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weightdecay_rate', type=float, default=0.0001, help='weightdecay rate')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
parser.add_argument('--eval_interval', type=int, default=1000, help='evaluation interval')
parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
args = parser.parse_args()


logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)
logging.info('batchsize={}'.format(args.batchsize))
logging.info('MomentumSGD(lr={})'.format(args.lr))
logging.info('WeightDecay(rate={})'.format(args.weightdecay_rate))

if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = TreePolicy()
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightdecay_rate, nesterov=True)
cross_entropy_loss = torch.nn.CrossEntropyLoss()
if args.use_amp:
    logging.info('use amp')
    scaler = torch.cuda.amp.GradScaler()

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    checkpoint = torch.load(args.resume)
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

# mini batch
def mini_batch(hcpevec):
    features1 = np.empty((len(hcpevec), FEATURES1_NUM * 9 * 9), dtype=np.float32)
    features2 = np.empty((len(hcpevec), FEATURES2_NUM, 9 * 9), dtype=np.float32)
    move = np.empty((len(hcpevec)), dtype=np.int32)

    cppshogi.hcpe_decode_with_move(hcpevec, features1, features2, move)

    features = np.concatenate([features1, features2[:,:,0].squeeze()], axis=1)

    return (torch.tensor(features).to(device),
            torch.tensor(move.astype(np.int64)).to(device)
            )

def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)

# train
itr = 0
sum_loss = 0
eval_interval = args.eval_interval
for e in range(args.epoch):
    np.random.shuffle(train_data)

    itr_epoch = 0
    sum_loss_epoch = 0
    for i in range(0, len(train_data) - args.batchsize + 1, args.batchsize):
        if args.use_amp:
            amp_context = torch.cuda.amp.autocast()
            amp_context.__enter__()

        model.train()

        x1, t1 = mini_batch(train_data[i:i+args.batchsize])
        y1 = model(x1)

        model.zero_grad()
        loss = cross_entropy_loss(y1, t1)

        if args.use_amp:
            amp_context.__exit__()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        t += 1
        itr += 1
        sum_loss += loss.item()
        itr_epoch += 1
        sum_loss_epoch += loss.item()

        # print train loss
        if t % eval_interval == 0:
            model.eval()

            x1, t1 = mini_batch(np.random.choice(test_data, args.testbatchsize))
            with torch.no_grad():
                y1 = model(x1)

                loss = cross_entropy_loss(y1, t1)

                logging.info('epoch = {}, iteration = {}, loss = {:.08f}, test loss = {:.08f}, test accuracy = {:.08f}'.format(
                    epoch + 1, t,
                    sum_loss / itr,
                    loss.item(),
                    accuracy(y1, t1)))
            itr = 0
            sum_loss = 0

    if args.use_amp:
        amp_context = torch.cuda.amp.autocast()
        amp_context.__enter__()

    if args.use_amp:
        amp_context.__exit__()

    # print train loss for each epoch
    itr_test = 0
    sum_test_loss = 0
    sum_test_accuracy = 0
    sum_test_entropy = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_data) - args.testbatchsize, args.testbatchsize):
            x1, t1 = mini_batch(test_data[i:i+args.testbatchsize])
            y1 = model(x1)

            itr_test += 1
            loss = cross_entropy_loss(y1, t1)
            sum_test_loss += loss.item()
            sum_test_accuracy += accuracy(y1, t1)

            entropy = (- F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1)
            sum_test_entropy += entropy.mean().item()

        logging.info('epoch = {}, iteration = {}, train loss avr = {:.08f}, test_loss = {:.08f}, test accuracy = {:.08f}, test entropy = {:.08f}'.format(
            epoch + 1, t,
            sum_loss_epoch / itr_epoch,
            sum_test_loss / itr_test,
            sum_test_accuracy / itr_test,
            sum_test_entropy / itr_test))

    epoch += 1

print('save the model')
serializers.save_npz(args.model, model)
print('save the optimizer')
state = {
    'epoch': epoch,
    't': t,
    'optimizer_state_dict': optimizer.state_dict(),
    }
if args.use_amp:
    state['scaler_state_dict'] = scaler.state_dict()
torch.save(state, args.state)
