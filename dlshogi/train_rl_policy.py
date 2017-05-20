import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from dlshogi.policy_network import *
from dlshogi.common import *
from dlshogi.ai import ProbabilisticPolicyPlayer

import shogi

import argparse
import random
import copy
import subprocess
import re
import math
import os

import logging

usiengine = r'E:/game/shogi/YaneuraOu/YaneuraOu-2017-early-sse42.exe'
usiengine_options = [
    ('USI_Ponder', 'false'),
    ('USI_Hash', '256'),
    ('Threads', '4'),
    ('Hash', '16'),
    ('MultiPV', '1'),
    ('WriteDebugLog', 'false'),
    ('NetworkDelay', '120'),
    ('NetworkDelay2', '1120'),
    ('MinimumThinkingTime', '2000'),
    ('MaxMovesToDraw', '0'),
    ('Contempt', '0'),
    ('EnteringKingRule', 'CSARule27'),
    ('EvalDir', 'eval'),
    ('EvalShare', 'true'),
    ('NarrowBook', 'false'),
    ('BookMoves', '16'),
    ('BookFile', 'no_book'),
    ('BookEvalDiff', '30'),
    ('BookEvalBlackLimit', '0'),
    ('BookEvalWhiteLimit', '-140'),
    ('BookDepthLimit', '16'),
    ('BookOnTheFly', 'false'),
    ('ConsiderBookMoveCount', 'false'),
    ('PvInterval', '300'),
    ('ResignValue', '99999'),
    ('nodestime', '0'),
    ('Param1', '0'),
    ('Param2', '0'),
    ('EvalSaveDir', 'evalsave'),
]
ptn_score = re.compile(r'score (cp|mate) (-{0,1}\d+)')

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

# start usi engine
proc_usiengine = subprocess.Popen(usiengine, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, cwd=os.path.dirname(usiengine))
proc_usiengine.stdin.write('usi\n')
proc_usiengine.stdin.flush()
while proc_usiengine.stdout.readline().strip() != 'usiok':
    pass

# init usi engine
for (name, value) in usiengine_options:
    proc_usiengine.stdin.write('setoption name ' + name + ' value ' + value + '\n')
    proc_usiengine.stdin.flush()
proc_usiengine.stdin.write('isready\n')
proc_usiengine.stdin.flush()
while proc_usiengine.stdout.readline().strip() != 'readyok':
    pass

def sigmoid_winning_rate(x):
    return 2.0 / (1.0 + math.exp(-x/600.0)) - 1.0

# value function by usi engine
def state_value(board):
    # usi engine
    proc_usiengine.stdin.write('position sfen ' + board.sfen() + '\n')
    proc_usiengine.stdin.flush()
    proc_usiengine.stdin.write('go byoyomi 100\n')
    proc_usiengine.stdin.flush()
    while True:
        usiengine_line = proc_usiengine.stdout.readline().strip()
        usiengine_cmd = usiengine_line.split(' ', 1)
        if usiengine_cmd[0] == 'bestmove':
            break
        elif usiengine_cmd[0] == 'info':
            usiengine_last_info = usiengine_cmd[1]
    # check score
    m = ptn_score.search(usiengine_last_info)
    if m:
        kind = m.group(1)
        score = int(m.group(2))
        if kind == 'mate':
            if score >= 0:
                return 1.0
            else:
                return -1.0
        else:
            return sigmoid_winning_rate(float(score))
    else:
        raise 

# see: https://github.com/Rochester-NRT/RocAlphaGo/blob/develop/AlphaGo/training/reinforcement_policy_trainer.py
def run_n_games(optimizer, learner, opponent, num_games):
    states = [shogi.Board() for _ in range(num_games)]
    learner_net = learner

    # Create one list of features (aka state tensors) and one of moves for each game being played.
    feature1_tensors = [[] for _ in range(num_games)]
    feature2_tensors = [[] for _ in range(num_games)]
    label_tensors = [[] for _ in range(num_games)]

    # List of booleans indicating whether the 'learner' player won.
    learner_won = [None] * num_games

    # Start all odd games with moves by 'opponent'. Even games will have 'learner' black.
    learner_color = [shogi.BLACK if i % 2 == 0 else shogi.WHITE for i in range(num_games)]
    odd_states = states[1::2]
    moves, labels = opponent.get_moves(odd_states)
    for st, mv in zip(odd_states, moves):
        st.push(mv)
    
    current = learner
    other = opponent
    idxs_to_unfinished_states = {i: states[i] for i in range(num_games)}
    win_count = 0
    move_number_sum = 0
    while len(idxs_to_unfinished_states) > 0:
        # Get next moves by current player for all unfinished states.
        moves, labels = current.get_moves(list(idxs_to_unfinished_states.values()))
        just_finished = []
        # Do each move to each state in order.
        for (idx, state), mv, label in zip(idxs_to_unfinished_states.items(), moves, labels):
            # Order is important here. We must get the training pair on the unmodified state before
            # updating it with do_move.
            if mv is None:
                just_finished.append(idx)
                move_number_sum += state.move_number
                continue

            value = state_value(state)

            if abs(value) > 0.70:
                learner_won[idx] = value * (1.0 if current is learner else -1.0)
                just_finished.append(idx)
                #print(idx, state.move_number, state.turn, learner_won[idx])
                move_number_sum += state.move_number
                if learner_won[idx] > 0:
                    win_count += 1
            else:
                if current is learner:
                    features1, features2 = make_input_features_from_board(state)
                    feature1_tensors[idx].append(features1)
                    feature2_tensors[idx].append(features2)
                    label_tensors[idx].append(label)
                state.push(mv)

        # Remove games that have finished from dict.
        for idx in just_finished:
            del idxs_to_unfinished_states[idx]

        # Swap 'current' and 'other' for next turn.
        current, other = other, current

    # Train on each game's results, setting the learning rate negative to 'unlearn' positions from
    # games where the learner lost.
    for st_tensor1, st_tensor2, label_tensor, won in zip(feature1_tensors, feature2_tensors, label_tensors, learner_won):
        if won is not None:
            x1 = Variable(cuda.to_gpu(np.array(st_tensor1, dtype=np.float32)))
            x2 = Variable(cuda.to_gpu(np.array(st_tensor2, dtype=np.float32)))
            t = Variable(cuda.to_gpu(np.array(label_tensor, dtype=np.int32)))

            y = model(x1, x2)

            model.cleargrads()
            loss = F.softmax_cross_entropy(y, t)
            loss.backward()

            optimizer.lr = alpha * won
            optimizer.update()

    # Return the win ratio.
    return float(win_count) / num_games, float(move_number_sum) / num_games

# list opponents
opponents = []
for file in os.listdir(args.out_directory):
    opponents.append(os.path.join(args.out_directory, file))
if len(opponents) == 0:
    opponents.append(args.initial_weights)

opponent_model = PolicyNetwork()
opponent_model.to_gpu()

player = ProbabilisticPolicyPlayer(model)
opponent = ProbabilisticPolicyPlayer(opponent_model)

logging.info('start training')

# start training
for i_iter in range(1, args.iterations + 1):
    # Randomly choose opponent from pool (possibly self), and playing
    # game_batch games against them.
    opp_path = np.random.choice(opponents)

    # Load new weights into opponent's network, but keep the same opponent object.
    serializers.load_npz(opp_path, opponent.model)

    # Run games (and learn from results). Keep track of the win ratio vs each opponent over
    # time.
    win_ratio, avr_move = run_n_games(optimizer, player, opponent, args.game_batch)
    logging.info('iterations = {}, games = {}, win_ratio = {}, avr_move = {}'.format(optimizer.epoch + 1, optimizer.t, win_ratio, avr_move))

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

# terminate usi engine
proc_usiengine.stdin.write('quit\n')
proc_usiengine.stdin.flush()
