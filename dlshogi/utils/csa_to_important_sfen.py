import argparse
from cshogi import Board, move_to_usi
from cshogi import CSA
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('sfen')
parser.add_argument('--margin', type=int, default=350)
parser.add_argument('--diff', type=int, default=400)
parser.add_argument('--win_names', nargs='*', default=[])
args = parser.parse_args()


def diff_sign(a, b, margin):
    if abs(a) < abs(b):
        return (a + margin) * b < 0 or (a - margin) * b < 0
    else:
        return (b + margin) * a < 0 or (b - margin) * a < 0


board = Board()

with open(args.sfen, 'w', newline='\n') as f:
    for filepath in glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True):
        for kif in CSA.Parser.parse_file(filepath):
            if kif.endgame not in ('%TORYO', '%KACHI'):
                continue
            if args.win_names and all(win_name not in kif.names[kif.win - 1] for win_name in args.win_names):
                continue
            # 評価値がない棋譜を除外
            if all(comment == '' for comment in kif.comments[0::2]) or all(comment == '' for comment in kif.comments[1::2]):
                continue

            board.reset()
            prev_score = 0
            is_important = False
            for i, (move, score) in enumerate(zip(kif.moves, kif.scores)):
                if not board.is_legal(move):
                    print("skip {}:{}:{}".format(filepath, i, move_to_usi(move)))
                    break

                board.push(move)

                if is_important:
                    # 重要な局面が見つかった場合、符号が一致するまで手を進める
                    if not diff_sign(score, prev_score, args.margin):
                        # 符号が一致した場合、それまでの手順を出力
                        f.write('startpos moves ' + ' '.join(move_to_usi(m) for m in board.history) + '\n')
                        break
                # 符号が異なり、かつ評価値の差がdiff以上の局面を抽出
                elif diff_sign(score, prev_score, args.margin) and abs(score - prev_score) > args.diff:
                    is_important = True

                prev_score = score
