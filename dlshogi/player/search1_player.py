import numpy as np

import os
import shogi

from dlshogi.common import *
from dlshogi.features import *
from dlshogi.network.value import *
from dlshogi.player.base_player import *

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.contrib import predictor

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
        self.modelfile = os.path.join(os.path.dirname(__file__), '../../model/model_value-best.hdf5')
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
            self.model = load_model(self.modelfile)
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

        # 自分の手番側の勝率にするため符号を反転
        x = np.array(features, dtype=np.float32)
        y = -self.model.predict(x)
        logits = y.reshape(-1)
        probabilities = (1.0 / (1.0 + np.exp(-y))).reshape(-1)

        for i, move in enumerate(legal_moves):
            # 勝率を表示
            print('info string {:5} : {:.5f}'.format(move.usi(), probabilities[i]))
            
        # 確率が最大の手を選ぶ(グリーディー戦略)
        selected_index = greedy(logits)
        # 確率に応じて手を選ぶ(ソフトマックス戦略)
        #selected_index = boltzmann(np.array(logits, dtype=np.float32), 0.5)
        bestmove = legal_moves[selected_index]

        print('bestmove', bestmove.usi())
