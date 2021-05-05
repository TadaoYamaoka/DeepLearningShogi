import numpy as np

# 移動の定数
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
] = range(20)

# 指し手を表すラベルの数
MAX_MOVE_LABEL_NUM = len(MOVE_DIRECTION) + 7 # 7はhand piece

MAX_PIECES_IN_HAND = [
    8, # 歩の持ち駒の上限
    4, 4, 4,
    4,
    2, 2,
]
MAX_PIECES_IN_HAND_SUM = sum(MAX_PIECES_IN_HAND)

# color
[BLACK, WHITE] = range(2)

# number of features
PIECETYPE_NUM = 14 # 駒の種類
MAX_ATTACK_NUM = 3 # 利き数の最大値
FEATURES1_NUM = 2 * (PIECETYPE_NUM + PIECETYPE_NUM + MAX_ATTACK_NUM)
FEATURES2_NUM = 2 * MAX_PIECES_IN_HAND_SUM + 1

HuffmanCodedPosAndEval = np.dtype([
    ('hcp', np.uint8, 32),
    ('eval', np.int16),
    ('bestMove16', np.uint16),
    ('gameResult', np.uint8),
    ('dummy', np.uint8),
    ])

HuffmanCodedPosAndEval2 = np.dtype([
    ('hcp', np.uint8, 32),
    ('eval', np.int16),
    ('bestMove16', np.uint16),
    ('result', np.uint8),
    ('dummy', np.uint8),
    ])
