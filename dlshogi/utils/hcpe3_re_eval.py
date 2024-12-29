import onnxruntime
import argparse
from tqdm import tqdm
import numpy as np
from cshogi import Board, dtypeHcp, dtypeMove16, dtypeEval
from cshogi.dlshogi import make_input_features, FEATURES1_NUM, FEATURES2_NUM

HuffmanCodedPosAndEval3 = np.dtype([
    ('hcp', dtypeHcp), # 開始局面
    ('moveNum', np.uint16), # 手数
    ('result', np.uint8), # 結果（xxxxxx11:勝敗、xxxxx1xx:千日手、xxxx1xxx:入玉宣言、xxx1xxxx:最大手数）
    ('opponent', np.uint8), # 対戦相手（0:自己対局、1:先手usi、2:後手usi）
    ])
MoveInfo = np.dtype([
    ('selectedMove16', dtypeMove16), # 指し手
    ('eval', dtypeEval), # 評価値
    ('candidateNum', np.uint16), # 候補手の数
    ])
MoveVisits = np.dtype([
    ('move16', dtypeMove16), # 候補手
    ('visitNum', np.uint16), # 訪問回数
    ])

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('hcpe3')
parser.add_argument('out_hcpe3')
parser.add_argument('--a', type=float, default=756.0864962951762)
parser.add_argument('--tensorrt', action='store_true')
args = parser.parse_args()

f = open(args.hcpe3, 'rb')
f_out = open(args.out_hcpe3, 'wb')

session = onnxruntime.InferenceSession(args.model, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] if args.tensorrt else ['CUDAExecutionProvider', 'CPUExecutionProvider'])

def value_to_score(values, a):
    scores = np.empty_like(values)
    scores[values == 1] = 30000
    scores[values == 0] = -30000
    mask = (values != 1) & (values != 0)
    scores[mask] = -a * np.log(1 / values[mask] - 1)
    return scores

board = Board()
with tqdm(desc="Processing") as pbar:
    while True:
        data = f.read(HuffmanCodedPosAndEval3.itemsize)
        if len(data) == 0:
            break
        hcpe = np.frombuffer(data, HuffmanCodedPosAndEval3, 1)[0]
        board.set_hcp(hcpe['hcp'])
        assert board.is_ok()
        move_num = hcpe['moveNum']
        f_out.write(data)

        move_info_list = []
        move_visits_list = []
        x1 = np.empty((move_num, FEATURES1_NUM, 9, 9), dtype=np.float32)
        x2 = np.empty((move_num, FEATURES2_NUM, 9, 9), dtype=np.float32)

        for i in range(move_num):
            move_info = np.frombuffer(f.read(MoveInfo.itemsize), MoveInfo, 1)[0]
            move_info_list.append(move_info.copy())
            make_input_features(board, x1[i], x2[i])
            candidate_num = move_info['candidateNum']
            move_visits = f.read(MoveVisits.itemsize * candidate_num)
            move_visits_list.append(move_visits)
            move = board.move_from_move16(move_info['selectedMove16'])
            board.push(move)
            assert board.is_ok()

        io_binding = session.io_binding()
        io_binding.bind_cpu_input('input1', x1)
        io_binding.bind_cpu_input('input2', x2)
        io_binding.bind_output('output_policy')
        io_binding.bind_output('output_value')
        session.run_with_iobinding(io_binding)
        _, values = io_binding.copy_outputs_to_cpu()

        scores = value_to_score(values.reshape(-1), args.a)
        for move_info, move_visits, score in zip(move_info_list, move_visits_list, scores):
            move_info['eval'] = score
            move_info.tofile(f_out)
            f_out.write(move_visits)
        pbar.update(1)
