import argparse
from email.policy import default
from cshogi import *
from cshogi import CSA
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('--diff', type=int, default=500)
parser.add_argument('--win_name')
parser.add_argument('--lose_sfen')
parser.add_argument('--aug_policy')
args = parser.parse_args()

if args.lose_sfen:
    lose_sfen = []
    if args.aug_policy:
        import onnxruntime
        import numpy as np
        from cshogi.dlshogi import make_input_features, make_move_label, FEATURES1_NUM, FEATURES2_NUM
        session = onnxruntime.InferenceSession(args.aug_policy, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        features1 = np.empty((1, FEATURES1_NUM, 9, 9), dtype=np.float32)
        features2 = np.empty((1, FEATURES2_NUM, 9, 9), dtype=np.float32)

board = Board()
for filepath in glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True):
    for kif in CSA.Parser.parse_file(filepath):
        if kif.endgame not in ('%TORYO', '%KACHI'):
            continue
        if args.win_name and kif.names[kif.win - 1] != args.win_name:
            continue
        # 評価値がない棋譜を除外
        if all(comment == '' for comment in kif.comments[0::2]) or all(comment == '' for comment in kif.comments[1::2]):
            continue

        board.reset()
        prev_score = 0
        for i, (move, score) in enumerate(zip(kif.moves, kif.scores)):
            if not board.is_legal(move):
                print("skip {}:{}:{}".format(filepath, i, move_to_usi(move)))
                break

            if score * prev_score < 0 and abs(score - prev_score) > args.diff:
                if (score > 0 and kif.win - 1 == board.turn) or (score < 0 and kif.win - 1 != board.turn):
                    if i % 2 != kif.win - 1:
                        j = i + 1
                    else:
                        j = i
                    if abs(kif.scores[j]) > 100:
                        base_score = int(kif.scores[j] * 0.9)
                        ok = True
                        for after_score in kif.scores[j::2]:
                            if (base_score > 0 and after_score < base_score) or (base_score < 0 and after_score > base_score):
                                ok = False
                                break
                        if ok:
                            print(filepath, board.sfen(), sep='\t')
                            if args.lose_sfen:
                                for k, (move, score) in enumerate(zip(kif.moves[i:], kif.scores[i:])):
                                    # 負けた方の手番
                                    if (i + k) % 2 != j % 2:
                                        if (base_score > 0 and score > 100) or (base_score < 0 and score < -100):
                                            break
                                        lose_sfen.append('startpos moves ' + ' '.join([move_to_usi(move) for move in board.history]) + '\n')

                                        # dlshogiのモデルで方策の確率が実際の指し手以上の手を追加で出力する
                                        if args.aug_policy:
                                            prev_move = board.pop()
                                            make_input_features(board, features1, features2)
                                            io_binding = session.io_binding()
                                            io_binding.bind_cpu_input('input1', features1)
                                            io_binding.bind_cpu_input('input2', features2)
                                            io_binding.bind_output('output_policy')
                                            io_binding.bind_output('output_value')
                                            session.run_with_iobinding(io_binding)
                                            y1, y2 = io_binding.copy_outputs_to_cpu()
                                            policy_logit = y1[0]
                                            prev_move_label = make_move_label(prev_move, board.turn)
                                            prev_move_policy_logit = policy_logit[prev_move_label]
                                            for legal_move in board.legal_moves:
                                                label = make_move_label(legal_move, board.turn)
                                                if policy_logit[label] > prev_move_policy_logit:
                                                    lose_sfen.append('startpos moves ' + ' '.join([move_to_usi(move) for move in board.history]) + ' ' + move_to_usi(legal_move) + '\n')
                                            board.push(prev_move)

                                    board.push(move)
                            break

            board.push(move)
            prev_score = score

if args.lose_sfen:
    with open(args.lose_sfen, 'w', newline='\n') as f:
        f.writelines(lose_sfen)
