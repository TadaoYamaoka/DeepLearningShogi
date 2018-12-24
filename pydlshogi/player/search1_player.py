import numpy as np
import chainer
from chainer import serializers
from chainer import cuda, Variable
import chainer.functions as F

import shogi

from pydlshogi.common import *
from pydlshogi.features import *
from pydlshogi.network.value import *
from pydlshogi.player.base_player import *

def greedy(logits):
    return np.argmax(logits)

def boltzmann(logits, temperature):
    logits /= temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return np.random.choice(len(logits), p=probabilities)

class Search1Player(BasePlayer):
    def __init__(self):
        super().__init__()
        self.modelfile = r'H:\src\python-dlshogi\model\model_value'
        self.model = None

    def usi(self):
        print('id name search1_player')
        print('option name modelfile type string default ' + self.modelfile)
        print('usiok')

    def setoption(self, option):
        if option[1] == 'modelfile':
            self.modelfile = option[3]

    def isready(self):
        if self.model is None:
            self.model = ValueNetwork()
            self.model.to_gpu()
        serializers.load_npz(self.modelfile, self.model)
        print('readyok')

    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return

        # 全ての合法手について
        legal_moves = []
        features = []
        for move in self.board.legal_moves:
            legal_moves.append(move)

            self.board.push(move) # 1手指す

            features.append(make_input_features_from_board(self.board))

            self.board.pop() # 1手戻す

        x = Variable(cuda.to_gpu(np.array(features, dtype=np.float32)))

        # 自分の手番側の勝率にするため符号を反転
        with chainer.no_backprop_mode():
            y = -self.model(x)

            logits = cuda.to_cpu(y.data).reshape(-1)
            probabilities = cuda.to_cpu(F.sigmoid(y).data).reshape(-1)

        for i, move in enumerate(legal_moves):
            # 勝率を表示
            print('info string {:5} : {:.5f}'.format(move.usi(), probabilities[i]))
            
        # 確率が最大の手を選ぶ(グリーディー戦略)
        selected_index = greedy(logits)
        # 確率に応じて手を選ぶ(ソフトマックス戦略)
        #selected_index = boltzmann(np.array(logits, dtype=np.float32), 0.5)
        bestmove = legal_moves[selected_index]

        print('bestmove', bestmove.usi())
