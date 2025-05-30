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
import os
import re
import importlib

import logging

def main(*argv):
    parser = argparse.ArgumentParser(description='Train policy value network')
    parser.add_argument('train_data', type=str, nargs='+', help='training data file')
    parser.add_argument('test_data', type=str, help='test data file')
    parser.add_argument('--batchsize', '-b', type=int, default=1024, help='Number of positions in each mini-batch')
    parser.add_argument('--testbatchsize', type=int, default=1024, help='Number of positions in each test mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
    parser.add_argument('--network', default='resnet10_swish', help='network type')
    parser.add_argument('--checkpoint', default='checkpoint-{epoch:03}.pth', help='checkpoint file name')
    parser.add_argument('--resume', '-r', default='', help='Resume from snapshot')
    parser.add_argument('--reset_optimizer', action='store_true')
    parser.add_argument('--model', type=str, help='model file name')
    parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file (for compatibility)')
    parser.add_argument('--log', help='log file path')
    parser.add_argument('--optimizer', default='SGD(momentum=0.9,nesterov=True)', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--lr_scheduler', help='learning rate scheduler')
    parser.add_argument('--scheduler_step_mode', type=str, default='epoch', choices=['epoch', 'step'], help='Scheduler step mode: epoch or step')
    parser.add_argument('--reset_scheduler', action='store_true')
    parser.add_argument('--clip_grad_max_norm', type=float, default=10.0, help='max norm of the gradients')
    parser.add_argument('--use_critic', action='store_true')
    parser.add_argument('--beta', type=float, help='entropy regularization coeff')
    parser.add_argument('--val_lambda', type=float, default=0.333, help='regularization factor')
    parser.add_argument('--val_lambda_decay_epoch', type=int, help='Number of total epochs to decay val_lambda to 0')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
    parser.add_argument('--eval_interval', type=int, default=1000, help='evaluation interval')
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--swa_start_epoch', type=int, default=1)
    parser.add_argument('--swa_freq', type=int, default=250)
    parser.add_argument('--swa_n_avr', type=int, default=10)
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--amp_dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='Data type for automatic mixed precision')
    parser.add_argument('--use_average', action='store_true')
    parser.add_argument('--use_evalfix', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--patch', type=str, help='Overwrite with the hcpe')
    parser.add_argument('--cache', type=str, help='training data cache file')
    args = parser.parse_args(argv)

    if args.log:
        logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', stream=sys.stdout, level=logging.DEBUG)
    logging.info('network {}'.format(args.network))
    logging.info('batchsize={}'.format(args.batchsize))
    logging.info('lr={}'.format(args.lr))
    logging.info('weight_decay={}'.format(args.weight_decay))
    if args.lr_scheduler:
        logging.info('lr_scheduler {}'.format(args.lr_scheduler))
    if args.use_critic:
        logging.info('use critic')
    if args.beta:
        logging.info('entropy regularization coeff={}'.format(args.beta))
    logging.info('val_lambda={}'.format(args.val_lambda))
    val_lambda = args.val_lambda

    if args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    model = policy_value_network(args.network)
    model.to(device)

    def create_optimizer(optimizer_str, model_params, lr, weight_decay):
        optimizer_name, optimizer_args = optimizer_str.split('(', 1)
        optimizer_args = eval(f'dict({optimizer_args.rstrip(")")})')
        if '.' in optimizer_name:
            module_name, class_name = optimizer_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            optimizer_class = getattr(module, class_name)
        else:
            optimizer_class = getattr(optim, optimizer_name)

        if weight_decay >= 0:
            optimizer_args["weight_decay"] = weight_decay

        optimizer = optimizer_class(model_params, lr=lr, **optimizer_args)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"Invalid optimizer type: {type(optimizer)}. Must be a subclass of torch.optim.Optimizer")
        return optimizer

    if args.optimizer[-1] != ')':
        args.optimizer += '()'
    optimizer = create_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    def create_scheduler(scheduler_str, optimizer):
        scheduler_name, scheduler_args = scheduler_str.split('(', 1)
        scheduler_args = eval(f'dict({scheduler_args.rstrip(")")})')
        if '.' in scheduler_name:
            module_name, class_name = scheduler_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            scheduler_class = getattr(module, class_name)
        else:
            scheduler_class = getattr(optim.lr_scheduler, scheduler_name)

        scheduler = scheduler_class(optimizer, **scheduler_args)
        if not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            raise TypeError(f"Invalid scheduler type: {type(scheduler)}. Must be a subclass of torch.optim.lr_scheduler.LRScheduler")
        return scheduler

    if args.lr_scheduler:
        scheduler = create_scheduler(args.lr_scheduler, optimizer)
    if args.use_swa:
        logging.info(f'use swa(swa_start_epoch={args.swa_start_epoch}, swa_freq={args.swa_freq}, swa_n_avr={args.swa_n_avr})')
        ema_a = args.swa_n_avr / (args.swa_n_avr + 1)
        ema_b = 1 / (args.swa_n_avr + 1)
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged : ema_a * averaged_model_parameter + ema_b * model_parameter
        swa_model = AveragedModel(model, avg_fn=ema_avg)
    def cross_entropy_loss_with_soft_target(pred, soft_targets):
        return torch.sum(-soft_targets * F.log_softmax(pred, dim=1), 1)
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
    if args.use_amp:
        logging.info(f'use amp dtype={args.amp_dtype}')
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
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
            if not args.reset_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
                if not args.lr_scheduler:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr
                        if args.weight_decay >= 0:
                            param_group['weight_decay'] = args.weight_decay
            if args.use_amp and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if args.lr_scheduler and not args.reset_scheduler and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            # for compatibility
            logging.info('Loading the optimizer state from {}'.format(args.resume))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.use_amp and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
        epoch = 0
        t = 0

    logging.info('optimizer {}'.format(re.sub(' +', ' ', str(optimizer).replace('\n', ''))))

    logging.info('Reading training data')
    train_len, actual_len = Hcpe3DataLoader.load_files(args.train_data, args.use_average, args.use_evalfix, args.temperature, args.patch, args.cache)
    train_data = np.arange(train_len, dtype=np.uint64)
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
                loss = loss1 + (1 - val_lambda) * loss2 + val_lambda * loss3
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

    def save_checkpoint():
        path = args.checkpoint.format(**{'epoch':epoch, 'step':t})
        logging.info('Saving the checkpoint to {}'.format(path))
        checkpoint = {
            'epoch': epoch,
            't': t,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()}
        if args.use_swa and epoch >= args.swa_start_epoch:
            checkpoint['swa_model'] = swa_model.state_dict()
        if args.lr_scheduler:
            checkpoint['scheduler'] = scheduler.state_dict()

        torch.save(checkpoint, path)

    # train
    steps = 0
    sum_loss1 = 0
    sum_loss2 = 0
    sum_loss3 = 0
    sum_loss = 0
    eval_interval = args.eval_interval
    for e in range(args.epoch):
        if args.lr_scheduler:
            logging.info('lr_scheduler lr={}'.format(scheduler.get_last_lr()[0]))
        if args.val_lambda_decay_epoch:
            # update val_lambda
            val_lambda = max(
                0,
                args.val_lambda * (1 - epoch / args.val_lambda_decay_epoch)
            )
            logging.info('update val_lambda={}'.format(val_lambda))
        epoch += 1
        steps_epoch = 0
        sum_loss1_epoch = 0
        sum_loss2_epoch = 0
        sum_loss3_epoch = 0
        sum_loss_epoch = 0
        for x1, x2, t1, t2, value in train_dataloader:
            t += 1
            steps += 1
            with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=amp_dtype):
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
                loss = loss1 + (1 - val_lambda) * loss2 + val_lambda * loss3

            scaler.scale(loss).backward()
            if args.clip_grad_max_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_max_norm)
            scaler.step(optimizer)
            scaler.update()

            if args.use_swa and epoch >= args.swa_start_epoch and t % args.swa_freq == 0:
                swa_model.update_parameters(model)

            sum_loss1 += loss1.item()
            sum_loss2 += loss2.item()
            sum_loss3 += loss3.item()
            sum_loss += loss.item()

            # print train loss
            if t % eval_interval == 0:
                model.eval()

                x1, x2, t1, t2, value = test_dataloader.sample()
                with torch.no_grad():
                    y1, y2 = model(x1, x2)

                    loss1 = cross_entropy_loss(y1, t1).mean()
                    loss2 = bce_with_logits_loss(y2, t2)
                    loss3 = bce_with_logits_loss(y2, value)
                    loss = loss1 + (1 - val_lambda) * loss2 + val_lambda * loss3

                    logging.info('epoch = {}, steps = {}, train loss = {:.07f}, {:.07f}, {:.07f}, {:.07f}, test loss = {:.07f}, {:.07f}, {:.07f}, {:.07f}, test accuracy = {:.07f}, {:.07f}'.format(
                        epoch, t,
                        sum_loss1 / steps, sum_loss2 / steps, sum_loss3 / steps, sum_loss / steps,
                        loss1.item(), loss2.item(), loss3.item(), loss.item(),
                        accuracy(y1, t1), binary_accuracy(y2, t2)))

                steps_epoch += steps
                sum_loss1_epoch += sum_loss1
                sum_loss2_epoch += sum_loss2
                sum_loss3_epoch += sum_loss3
                sum_loss_epoch += sum_loss

                steps = 0
                sum_loss1 = 0
                sum_loss2 = 0
                sum_loss3 = 0
                sum_loss = 0

            if args.lr_scheduler and args.scheduler_step_mode == 'step':
                scheduler.step()

        steps_epoch += steps
        sum_loss1_epoch += sum_loss1
        sum_loss2_epoch += sum_loss2
        sum_loss3_epoch += sum_loss3
        sum_loss_epoch += sum_loss

        # print train loss and test loss for each epoch
        test_loss1, test_loss2, test_loss3, test_loss, test_accuracy1, test_accuracy2, test_entropy1, test_entropy2 = test(model)

        logging.info('epoch = {}, steps = {}, train loss avr = {:.07f}, {:.07f}, {:.07f}, {:.07f}, test loss = {:.07f}, {:.07f}, {:.07f}, {:.07f}, test accuracy = {:.07f}, {:.07f}, test entropy = {:.07f}, {:.07f}'.format(
            epoch, t,
            sum_loss1_epoch / steps_epoch, sum_loss2_epoch / steps_epoch, sum_loss3_epoch / steps_epoch, sum_loss_epoch / steps_epoch,
            test_loss1, test_loss2, test_loss3, test_loss,
            test_accuracy1, test_accuracy2,
            test_entropy1, test_entropy2))

        if args.lr_scheduler and args.scheduler_step_mode == 'epoch':
            scheduler.step()

        # save checkpoint
        if args.checkpoint:
            save_checkpoint()

    # save model
    if args.model:
        if args.use_swa and epoch >= args.swa_start_epoch:
            logging.info('Updating batch normalization')
            forward_ = swa_model.forward
            swa_model.forward = lambda x : forward_(**x)
            with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=amp_dtype):
                update_bn(hcpe_loader(train_data, args.batchsize), swa_model)
            del swa_model.forward

            # print test loss with swa model
            test_loss1, test_loss2, test_loss3, test_loss, test_accuracy1, test_accuracy2, test_entropy1, test_entropy2 = test(swa_model)

            logging.info('epoch = {}, steps = {}, swa test loss = {:.07f}, {:.07f}, {:.07f}, {:.07f}, swa test accuracy = {:.07f}, {:.07f}, swa test entropy = {:.07f}, {:.07f}'.format(
                epoch, t,
                test_loss1, test_loss2, test_loss3, test_loss,
                test_accuracy1, test_accuracy2,
                test_entropy1, test_entropy2))

        model_path = args.model.format(**{'epoch':epoch, 'step':t})
        logging.info('Saving the model to {}'.format(model_path))
        serializers.save_npz(model_path, swa_model.module if args.use_swa else model)

if __name__ == '__main__':
    main(*sys.argv[1:])
