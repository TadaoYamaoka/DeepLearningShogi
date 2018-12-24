import numpy as np

import os
import shogi

from dlshogi.common import *
from dlshogi.features import *
from dlshogi.network.policy import *
from dlshogi.player.base_player import *

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.contrib import predictor

def greedy(logits):
    return logits.index(max(logits))

def boltzmann(logits, temperature):
    logits /= temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return np.random.choice(len(logits), p=probabilities)

class PolicyPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        self.modelfile = os.path.join(os.path.dirname(__file__), '../../model/model_policy-best.hdf5')
        self.model = None

    def usi(self):
        print('id name policy_player')
        print('option name modelfile type string default ' + self.modelfile)
        print('usiok')

    def setoption(self, option):
        if option[1] == 'modelfile':
            self.modelfile = option[3]

    def isready(self):
        if self.model is None:
            self.model = load_model(self.modelfile, custom_objects={'Bias': Bias})
            layer_name = 'policy_head_notop'
            self.intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
            self.intermediate_layer_model.summary()
        print('readyok')

    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return

        features = make_input_features_from_board(self.board)
        
        x = np.array([features], dtype=np.float32)
        y = self.intermediate_layer_model.predict(x)
        logits = y[0]
        odds = np.exp(logits)
        probabilities = odds / np.sum(odds)

        # 全ての合法手について
        legal_moves = []
        legal_logits = []
        for move in self.board.legal_moves:
            # ラベルに変換
            label = make_output_label(move, self.board.turn)
            # 合法手とその指し手の確率(logits)を格納
            legal_moves.append(move)
            legal_logits.append(logits[label])
            # 確率を表示
            print('info string {:5} : {:.5f}'.format(move.usi(), probabilities[label]))
            
        # 確率が最大の手を選ぶ(グリーディー戦略)
        selected_index = greedy(legal_logits)
        # 確率に応じて手を選ぶ(ソフトマックス戦略)
        #selected_index = boltzmann(np.array(legal_logits, dtype=np.float32), 0.5)
        bestmove = legal_moves[selected_index]

        print('bestmove', bestmove.usi())
