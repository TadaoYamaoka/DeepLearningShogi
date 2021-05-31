import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, update_bn

from dlshogi.common import *
from dlshogi.network.policy_value_network import policy_value_network
from dlshogi import serializers
from dlshogi.data_loader import Hcpe3DataLoader
from dlshogi.data_loader import DataLoader


import argparse
import random
import sys

import logging

def main(*args):
    parser = argparse.ArgumentParser(description='Train policy value network')
    parser.add_argument('train_data', type=str, nargs='+', help='training data file')
    parser.add_argument('test_data', type=str, help='test data file')
    parser.add_argument('--batchsize', '-b', type=int, default=1024, help='Number of positions in each mini-batch')
    parser.add_argument('--testbatchsize', type=int, default=640, help='Number of positions in each test mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
    parser.add_argument('--network', type=str, default='resnet10_swish', choices=['wideresnet10', 'wideresnet15', 'resnet10_swish', 'resnet15_swish', 'resnet20_swish', 'senet10', 'senet10_swish', 'senet15_swish', 'senet20_swish'], help='network type')
    parser.add_argument('--user_network', type=str)
    parser.add_argument('--checkpoint', type=str, help='checkpoint file name')
    parser.add_argument('--resume', '-r', default='', help='Resume from snapshot')
    parser.add_argument('--model', type=str, help='model file name')
    parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file (for compatibility)')
    parser.add_argument('--log', default=None, help='log file path')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weightdecay_rate', type=float, default=0.0001, help='weightdecay rate')
    parser.add_argument('--clip_grad_max_norm', type=float, default=10.0, help='max norm of the gradients')
    parser.add_argument('--use_critic', action='store_true')
    parser.add_argument('--beta', type=float, help='entropy regularization coeff')
    parser.add_argument('--val_lambda', type=float, default=0.333, help='regularization factor')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
    parser.add_argument('--eval_interval', type=int, default=1000, help='evaluation interval')
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--swa_freq', type=int, default=250)
    parser.add_argument('--swa_n_avr', type=int, default=10)
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--use_average', action='store_true')
    parser.add_argument('--use_evalfix', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args(args)

    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)
    logging.info('batchsize={}'.format(args.batchsize))
    logging.info('MomentumSGD(lr={})'.format(args.lr))
    logging.info('WeightDecay(rate={})'.format(args.weightdecay_rate))
    if args.use_critic:
        logging.info('use critic')
    if args.beta:
        logging.info('entropy regularization coeff={}'.format(args.beta))
    logging.info('val_lambda={}'.format(args.val_lambda))

    if args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    model = policy_value_network(args.user_network if args.user_network else args.network)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightdecay_rate, nesterov=True)
    if args.use_swa:
        logging.info(f'use swa(swa_freq={args.swa_freq}, swa_n_avr={args.swa_n_avr})')
        ema_a = args.swa_n_avr / (args.swa_n_avr + 1)
        ema_b = 1 / (args.swa_n_avr + 1)
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged : ema_a * averaged_model_parameter + ema_b * model_parameter
        swa_model = AveragedModel(model, avg_fn=ema_avg)
    def cross_entropy_loss_with_soft_target(pred, soft_targets):
        return torch.sum(-soft_targets * F.log_softmax(pred, dim=1), 1)
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
    if args.use_amp:
        logging.info('use amp')
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    if args.use_evalfix:
        logging.info('use evalfix')
    logging.info('temperature={}'.format(args.temperature))

    # Init/Resume
    if args.initmodel:
        # for compatibility
        logging.info('Loading the model from {}'.format(args.initmodel))
        serializers.load_npz(args.initmodel, model)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        epoch = checkpoint['epoch']
        t = checkpoint['t']
        if 'model' in checkpoint:
            logging.info('Loading the checkpoint from {}'.format(args.resume))
            model.load_state_dict(checkpoint['model'])
            if args.use_swa and 'swa_model' in checkpoint:
                swa_model.load_state_dict(checkpoint['swa_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.use_amp and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
        else:
            # for compatibility
            logging.info('Loading the optimizer state from {}'.format(args.resume))
            base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.use_amp and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
        epoch = 0
        t = 0

    logging.info('Reading training data')
    train_len, actual_len = Hcpe3DataLoader.load_files(args.train_data, args.use_average, args.use_evalfix, args.temperature)
    train_data = np.arange(train_len, dtype=np.int32)
    logging.info('Reading test data')
    test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)

    if args.use_average:
        logging.info('train position num before preprocessing = {}'.format(actual_len))
    logging.info('train position num = {}'.format(len(train_data)))
    logging.info('test position num = {}'.format(len(test_data)))

    train_dataloader = Hcpe3DataLoader(train_data, args.batchsize, device, shuffle=True)
    test_dataloader = DataLoader(test_data, args.testbatchsize, device)

    # for SWA update_bn
    def hcpe_loader(data, batchsize):
        for x1, x2, t1, t2, value in Hcpe3DataLoader(data, batchsize, device):
            yield { 'x1':x1, 'x2':x2 }

    def accuracy(y, t):
        return (torch.max(y, 1)[1] == t).sum().item() / len(t)

    def binary_accuracy(y, t):
        pred = y >= 0
        truth = t >= 0.5
        return pred.eq(truth).sum().item() / len(t)

    def test(model):
        steps = 0
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

                steps += 1
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

        return (sum_test_loss1 / steps,
                sum_test_loss2 / steps,
                sum_test_loss3 / steps,
                sum_test_loss / steps,
                sum_test_accuracy1 / steps,
                sum_test_accuracy2 / steps,
                sum_test_entropy1 / steps,
                sum_test_entropy2 / steps)

    # train
    steps = 0
    sum_loss1 = 0
    sum_loss2 = 0
    sum_loss3 = 0
    sum_loss = 0
    eval_interval = args.eval_interval
    for e in range(args.epoch):
        steps_epoch = 0
        sum_loss1_epoch = 0
        sum_loss2_epoch = 0
        sum_loss3_epoch = 0
        sum_loss_epoch = 0
        for x1, x2, t1, t2, value in train_dataloader:
            t += 1
            steps += 1
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                model.train()

                y1, y2 = model(x1, x2)

                model.zero_grad()
                loss1 = cross_entropy_loss_with_soft_target(y1, t1)
                if args.use_critic:
                    z = t2.view(-1) - value.view(-1) + 0.5
                    loss1 = (loss1 * z).mean()
                else:
                    loss1 = loss1.mean()
                if args.beta:
                    loss1 += args.beta * (F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1).mean()
                loss2 = bce_with_logits_loss(y2, t2)
                loss3 = bce_with_logits_loss(y2, value)
                loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3

            scaler.scale(loss).backward()
            if args.clip_grad_max_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_max_norm)
            scaler.step(optimizer)
            scaler.update()

            if args.use_swa and t % args.swa_freq == 0:
                swa_model.update_parameters(model)

            sum_loss1 += loss1.item()
            sum_loss2 += loss2.item()
            sum_loss3 += loss3.item()
            sum_loss += loss.item()
            steps_epoch += 1
            sum_loss1_epoch += loss1.item()
            sum_loss2_epoch += loss2.item()
            sum_loss3_epoch += loss3.item()
            sum_loss_epoch += loss.item()

            # print train loss
            if t % eval_interval == 0:
                model.eval()

                x1, x2, t1, t2, value = test_dataloader.sample()
                with torch.no_grad():
                    y1, y2 = model(x1, x2)

                    loss1 = cross_entropy_loss(y1, t1).mean()
                    loss2 = bce_with_logits_loss(y2, t2)
                    loss3 = bce_with_logits_loss(y2, value)
                    loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3

                    logging.info('epoch = {}, steps = {}, loss = {:.06f}, {:.06f}, {:.06f}, {:.06f}, test loss = {:.06f}, {:.06f}, {:.06f}, {:.06f}, test accuracy = {:.06f}, {:.06f}'.format(
                        epoch, t,
                        sum_loss1 / steps, sum_loss2 / steps, sum_loss3 / steps, sum_loss / steps,
                        loss1.item(), loss2.item(), loss3.item(), loss.item(),
                        accuracy(y1, t1), binary_accuracy(y2, t2)))
                steps = 0
                sum_loss1 = 0
                sum_loss2 = 0
                sum_loss3 = 0
                sum_loss = 0

        # print train loss and test loss for each epoch
        test_loss1, test_loss2, test_loss3, test_loss, test_accuracy1, test_accuracy2, test_entropy1, test_entropy2 = test(model)

        logging.info('epoch = {}, steps = {}, train loss avr = {:.06f}, {:.06f}, {:.06f}, {:.06f}, test loss = {:.06f}, {:.06f}, {:.06f}, {:.06f}, test accuracy = {:.06f}, {:.06f}, test entropy = {:.06f}, {:.06f}'.format(
            epoch, t,
            sum_loss1_epoch / steps_epoch, sum_loss2_epoch / steps_epoch, sum_loss3_epoch / steps_epoch, sum_loss_epoch / steps_epoch,
            test_loss1, test_loss2, test_loss3, test_loss,
            test_accuracy1, test_accuracy2,
            test_entropy1, test_entropy2))

        epoch += 1

    if args.checkpoint:
        logging.info('Saving the checkpoint to {}'.format(args.checkpoint))
        checkpoint = {
            'epoch': epoch,
            't': t,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()}
        if args.use_swa:
            checkpoint['swa_model'] = swa_model.state_dict()

        torch.save(checkpoint, args.checkpoint)


    # save model
    if args.model:
        if args.use_swa:
            logging.info('Updating batch normalization')
            forward_ = swa_model.forward
            swa_model.forward = lambda x : forward_(**x)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                update_bn(hcpe_loader(train_data, args.batchsize), swa_model)
            del swa_model.forward

            # print test loss with swa model
            test_loss1, test_loss2, test_loss3, test_loss, test_accuracy1, test_accuracy2, test_entropy1, test_entropy2 = test(swa_model)

            logging.info('epoch = {}, steps = {}, swa test loss = {:.06f}, {:.06f}, {:.06f}, {:.06f}, swa test accuracy = {:.06f}, {:.06f}, swa test entropy = {:.06f}, {:.06f}'.format(
                epoch, t,
                test_loss1, test_loss2, test_loss3, test_loss,
                test_accuracy1, test_accuracy2,
                test_entropy1, test_entropy2))

        logging.info('Saving the model to {}'.format(args.model))
        serializers.save_npz(args.model, swa_model.module if args.use_swa else model)

if __name__ == '__main__':
    main(*sys.argv[1:])
