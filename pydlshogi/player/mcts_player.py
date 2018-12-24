import numpy as np
import chainer
from chainer import serializers
from chainer import cuda, Variable
import chainer.functions as F

import shogi

from pydlshogi.common import *
from pydlshogi.features import *
from pydlshogi.network.policy_value_resnet import *
from pydlshogi.player.base_player import *
from pydlshogi.uct.uct_node import *

import math
import time
import copy

# UCBのボーナス項の定数
C_PUCT = 1.0
# 1手当たりのプレイアウト数
CONST_PLAYOUT = 300
# 投了する勝率の閾値
RESIGN_THRESHOLD = 0.01
# 温度パラメータ
TEMPERATURE = 1.0

def softmax_temperature_with_normalize(logits, temperature):
    # 温度パラメータを適用
    logits /= temperature

    # 確率を計算(オーバーフローを防止するため最大値で引く)
    max_logit = max(logits)
    probabilities = np.exp(logits - max_logit)

    # 合計が1になるように正規化
    sum_probabilities = sum(probabilities)
    probabilities /= sum_probabilities

    return probabilities

class PlayoutInfo:
    def __init__(self):
        self.halt = 0 # 探索を打ち切る回数
        self.count = 0 # 現在の探索回数

class MCTSPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        # モデルファイルのパス
        self.modelfile = r'H:\src\python-dlshogi\model\model_policy_value_resnet'
        self.model = None # モデル

        # ノードの情報
        self.node_hash = NodeHash()
        self.uct_node = [UctNode() for _ in range(UCT_HASH_SIZE)]

        # プレイアウト回数管理
        self.po_info = PlayoutInfo()
        self.playout = CONST_PLAYOUT

        # 温度パラメータ
        self.temperature = TEMPERATURE

    # UCB値が最大の手を求める
    def select_max_ucb_child(self, board, current_node):
        child_num = current_node.child_num
        child_win = current_node.child_win
        child_move_count = current_node.child_move_count

        q = np.divide(child_win, child_move_count, out=np.repeat(np.float32(0.5), child_num), where=child_move_count != 0)
        u = np.sqrt(np.float32(current_node.move_count)) / (1 + child_move_count)
        ucb = q + C_PUCT * current_node.nnrate * u

        return np.argmax(ucb)


    # ノードの展開
    def expand_node(self, board):
        index = self.node_hash.find_same_hash_index(board.zobrist_hash(), board.turn, board.move_number)

        # 合流先が検知できれば, それを返す
        if not index == UCT_HASH_SIZE:
            return index
    
        # 空のインデックスを探す
        index = self.node_hash.search_empty_index(board.zobrist_hash(), board.turn, board.move_number)

        # 現在のノードの初期化
        current_node = self.uct_node[index]
        current_node.move_count = 0
        current_node.win = 0.0
        current_node.child_num = 0
        current_node.evaled = False
        current_node.value_win = 0.0

        # 候補手の展開
        current_node.child_move = [move for move in board.legal_moves]
        child_num = len(current_node.child_move)
        current_node.child_index = [NOT_EXPANDED for _ in range(child_num)]
        current_node.child_move_count = np.zeros(child_num, dtype=np.int32)
        current_node.child_win = np.zeros(child_num, dtype=np.float32)

        # 子ノードの個数を設定
        current_node.child_num = child_num

        # ノードを評価
        if child_num > 0:
            self.eval_node(board, index)
        else:
            current_node.value_win = 0.0
            current_node.evaled = True

        return index

    # 探索を打ち切るか確認
    def interruption_check(self):
        child_num = self.uct_node[self.current_root].child_num
        child_move_count = self.uct_node[self.current_root].child_move_count
        rest = self.po_info.halt - self.po_info.count

        # 探索回数が最も多い手と次に多い手を求める
        second, first = child_move_count[np.argpartition(child_move_count, -2)[-2:]]

        # 残りの探索を全て次善手に費やしても最善手を超えられない場合は探索を打ち切る
        if first - second > rest:
            return True
        else:
            return False

    # UCT探索
    def uct_search(self, board, current):
        current_node = self.uct_node[current]

        # 詰みのチェック
        if current_node.child_num == 0:
            return 1.0 # 反転して値を返すため1を返す

        child_move = current_node.child_move
        child_move_count = current_node.child_move_count
        child_index = current_node.child_index

        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(board, current_node)
        # 選んだ手を着手
        board.push(child_move[next_index])

        # ノードの展開の確認
        if child_index[next_index] == NOT_EXPANDED:
            # ノードの展開(ノード展開処理の中でノードを評価する)
            index = self.expand_node(board)
            child_index[next_index] = index
            child_node = self.uct_node[index]

            # valueを勝敗として返す
            result = 1 - child_node.value_win
        else:
            # 手番を入れ替えて1手深く読む
            result = self.uct_search(board, child_index[next_index])

        # 探索結果の反映
        current_node.win += result
        current_node.move_count += 1
        current_node.child_win[next_index] += result
        current_node.child_move_count[next_index] += 1

        # 手を戻す
        board.pop()

        return 1 - result

    # ノードを評価
    def eval_node(self, board, index):
        eval_features = [make_input_features_from_board(board)]

        x = Variable(cuda.to_gpu(np.array(eval_features, dtype=np.float32)))
        with chainer.no_backprop_mode():
            y1, y2 = self.model(x)

            logits = cuda.to_cpu(y1.data)[0]
            value = cuda.to_cpu(F.sigmoid(y2).data)[0]

        current_node = self.uct_node[index]
        child_num = current_node.child_num
        child_move = current_node.child_move
        color = self.node_hash[index].color

        # 合法手でフィルター
        legal_move_labels = []
        for i in range(child_num):
            legal_move_labels.append(make_output_label(child_move[i], color))

        # Boltzmann分布
        probabilities = softmax_temperature_with_normalize(logits[legal_move_labels], self.temperature)

        # ノードの値を更新
        current_node.nnrate = probabilities
        current_node.value_win = float(value)
        current_node.evaled = True

    def usi(self):
        print('id name mcts_player')
        print('option name modelfile type string default ' + self.modelfile)
        print('option name playout type spin default ' + str(self.playout) + ' min 100 max 10000')
        print('option name temperature type spin default ' + str(int(self.temperature * 100)) + ' min 10 max 1000')
        print('usiok')

    def setoption(self, option):
        if option[1] == 'modelfile':
            self.modelfile = option[3]
        elif option[1] == 'playout':
            self.playout = int(option[3])
        elif option[1] == 'temperature':
            self.temperature = int(option[3]) / 100

    def isready(self):
        # モデルをロード
        if self.model is None:
            self.model = PolicyValueResnet()
            self.model.to_gpu()
        serializers.load_npz(self.modelfile, self.model)
        # ハッシュを初期化
        self.node_hash.initialize()
        print('readyok')

    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return

        # 探索情報をクリア
        self.po_info.count = 0

        # 古いハッシュを削除
        self.node_hash.delete_old_hash(self.board, self.uct_node)

        # 探索開始時刻の記録
        begin_time = time.time()

        # 探索回数の閾値を設定
        self.po_info.halt = self.playout

        # ルートノードの展開
        self.current_root = self.expand_node(self.board)

        # 候補手が1つの場合は、その手を返す
        current_node = self.uct_node[self.current_root]
        child_num = current_node.child_num
        child_move = current_node.child_move
        if child_num == 1:
            print('bestmove', child_move[0].usi())
            return

        # プレイアウトを繰り返す
        # 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
        while self.po_info.count < self.po_info.halt:
            # 探索回数を1回増やす
            self.po_info.count += 1
            # 1回プレイアウトする
            self.uct_search(self.board, self.current_root)
            # 探索を打ち切るか確認
            if self.interruption_check() or not self.node_hash.enough_size:
                break

        # 探索にかかった時間を求める
        finish_time = time.time() - begin_time

        child_move_count = current_node.child_move_count
        if self.board.move_number < 10:
            # 訪問回数に応じた確率で手を選択する
            selected_index = np.random.choice(np.arange(child_num), p=child_move_count/sum(child_move_count))
        else:
            # 訪問回数最大の手を選択する
            selected_index = np.argmax(child_move_count)

        child_win = current_node.child_win

        # for debug
        for i in range(child_num):
            print('{:3}:{:5} move_count:{:4} nn_rate:{:.5f} win_rate:{:.5f}'.format(
                i, child_move[i].usi(), child_move_count[i],
                current_node.nnrate[i],
                child_win[i] / child_move_count[i] if child_move_count[i] > 0 else 0))

        # 選択した着手の勝率の算出
        best_wp = child_win[selected_index] / child_move_count[selected_index]

        # 閾値未満の場合投了
        if best_wp < RESIGN_THRESHOLD:
            print('bestmove resign')
            return

        bestmove = child_move[selected_index]

        # 勝率を評価値に変換
        if best_wp == 1.0:
            cp = 30000
        else:
            cp = int(-math.log(1.0 / best_wp - 1.0) * 600)

        print('info nps {} time {} nodes {} hashfull {} score cp {} pv {}'.format(
            int(current_node.move_count / finish_time),
            int(finish_time * 1000),
            current_node.move_count,
            int(self.node_hash.get_usage_rate() * 1000),
            cp, bestmove.usi()))

        print('bestmove', bestmove.usi())
