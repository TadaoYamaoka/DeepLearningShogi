import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from dlshogi.policy_network import *
from dlshogi.common import *

import cppshogi

import argparse
import random
import os

import logging

parser = argparse.ArgumentParser(description='Traning RL policy network')
parser.add_argument('initial_weights', help='Path to model file with inital weights (i.e. result of supervised training).')
parser.add_argument('out_directory', help='Path to folder where the model params and metadata will be saved after each epoch.')
parser.add_argument('--game-batch', '-b', help='Number of games per mini-batch', type=int, default=20)
parser.add_argument('--iterations', '-i', help='Number of training batches/iterations', type=int, default=10000)
parser.add_argument('--model', type=str, default='model_rl', help='model file name')
parser.add_argument('--state', type=str, default='state_rl', help='state file name')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--save-every', help='Save policy as a new opponent every n batches', type=int, default=500)
parser.add_argument('--eval_dir', help='apery eval dir', type=str, default=r'H:\src\elmo_for_learn\bin\20161007')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

model = PolicyNetwork()
model.to_gpu()

alpha = args.lr
optimizer = optimizers.SGD(lr=alpha)
optimizer.use_cleargrads()
optimizer.setup(model)

# Init/Resume
print('Load model from', args.initial_weights)
serializers.load_npz(args.initial_weights, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

# init cppshogi
cppshogi.setup_eval_dir(args.eval_dir)
states = cppshogi.States(args.game_batch)

# see: https://github.com/Rochester-NRT/RocAlphaGo/blob/develop/AlphaGo/training/reinforcement_policy_trainer.py
def run_n_games(optimizer, learner, opponent, num_games):
    states.default_start_position()

    # Create one list of features (aka state tensors) and one of moves for each game being played.
    features1_tensors = [[] for _ in range(num_games)]
    features2_tensors = [[] for _ in range(num_games)]
    labels_tensors = [[] for _ in range(num_games)]
    values_tensors = [[] for _ in range(num_games)]

    # List of booleans indicating whether the 'learner' player won.
    learner_won = [None] * num_games

    # Start all odd games with moves by 'opponent'. Even games will have 'learner' black.
    learner_color = [BLACK if i % 2 == 0 else WHITE for i in range(num_games)]
    odd_features1 = np.empty((num_games, 2 * 14, 9, 9), dtype=np.float32)
    odd_features2 = np.empty((num_games, 2 * MAX_PIECES_IN_HAND_SUM + 1, 9, 9), dtype=np.float32)
    states.make_odd_input_features(odd_features1, odd_features2)
    x1 = Variable(cuda.to_gpu(odd_features1), volatile=True)
    x2 = Variable(cuda.to_gpu(odd_features2), volatile=True)
    y = opponent(x1, x2, test=True)
    y_data = cuda.to_cpu(y.data)
    states.do_odd_moves(y_data)

    current = learner
    other = opponent
    unfinished_states_num = num_games
    move_number_sum = 0
    while unfinished_states_num > 0:
        move_number_sum += unfinished_states_num

        # Get next moves by current player for all unfinished states.
        features1 = np.empty((unfinished_states_num, FEATURES1_NUM, 9, 9), dtype=np.float32)
        features2 = np.empty((unfinished_states_num, FEATURES2_NUM, 9, 9), dtype=np.float32)
        unfinished_list = states.make_unfinished_input_features(features1, features2)
        x1 = Variable(cuda.to_gpu(features1), volatile=True)
        x2 = Variable(cuda.to_gpu(features2), volatile=True)
        y = current(x1, x2, test=True)
        y_data = cuda.to_cpu(y.data)

        labels = np.empty((unfinished_states_num), dtype=np.int32)
        values = np.empty((unfinished_states_num), dtype=np.float32)
        unfinished_states_num = states.do_unfinished_moves_and_eval(current is learner, y_data, labels, values)

        # 特徴を保存
        if current is learner:
            for i, idx in enumerate(unfinished_list):
                features1_tensors[idx].append(features1[i])
                features2_tensors[idx].append(features2[i])
                labels_tensors[idx].append(labels[i])
                values_tensors[idx].append(values[i])

        # Swap 'current' and 'other' for next turn.
        current, other = other, current

    learner_won = np.empty(num_games, dtype=np.int32)
    states.get_learner_wons(learner_won)
        
    # Train on all game's results
    features1_tensor_all = []
    features2_tensor_all = []
    labels_tensor_all = []
    rewards_tensor_all = []
    for features1_tensor, features2_tensor, labels_tensor, values_tensor, won in zip(features1_tensors, features2_tensors, labels_tensors, values_tensors, learner_won.astype(np.float32)):
        features1_tensor_all.extend(features1_tensor)
        features2_tensor_all.extend(features2_tensor)
        labels_tensor_all.extend(labels_tensor)
        rewards_tensor_all.extend(list(won - np.array(values_tensor, dtype=np.float32)))

    x1 = Variable(cuda.to_gpu(np.array(features1_tensor_all, dtype=np.float32)))
    x2 = Variable(cuda.to_gpu(np.array(features2_tensor_all, dtype=np.float32)))
    t = Variable(cuda.to_gpu(np.array(labels_tensor_all, dtype=np.int32)))
    z = Variable(cuda.to_gpu(np.array(rewards_tensor_all, dtype=np.float32)))

    y = learner(x1, x2)

    learner.cleargrads()
    loss = F.mean(F.softmax_cross_entropy(y, t, reduce='no') * z)
    loss.backward()

    optimizer.update()

    # Return the win ratio.
    return np.average(learner_won), float(move_number_sum) / num_games, loss.data

# list opponents
opponents = []
for file in os.listdir(args.out_directory):
    opponents.append(os.path.join(args.out_directory, file))
if len(opponents) == 0:
    opponents.append(args.initial_weights)

opponent = PolicyNetwork()
opponent.to_gpu()

logging.info('start training')

# start training
for i_iter in range(1, args.iterations + 1):
    # Randomly choose opponent from pool (possibly self), and playing
    # game_batch games against them.
    opp_path = np.random.choice(opponents)

    # Load new weights into opponent's network, but keep the same opponent object.
    serializers.load_npz(opp_path, opponent)

    # Run games (and learn from results). Keep track of the win ratio vs each opponent over
    # time.
    win_ratio, avr_move, loss = run_n_games(optimizer, model, opponent, args.game_batch)
    logging.info('iterations = {}, win_ratio = {}, avr_move = {}, loss = {}'.format(optimizer.epoch + 1, win_ratio, avr_move, loss))

    optimizer.new_epoch()

    # Add player to batch of oppenents once in a while.
    if i_iter % args.save_every == 0:
        # Save models.
        player_model_path = os.path.join(args.out_directory, "model.%05d" % optimizer.epoch)
        serializers.save_npz(player_model_path, model)

        opponents.append(player_model_path)

logging.info('end training')

print('save the model')
serializers.save_npz(args.model + ".%05d" % optimizer.epoch, model)
print('save the optimizer')
serializers.save_npz(args.state + ".%05d" % optimizer.epoch, optimizer)
