import argparse
from email.policy import default
from cshogi import *
from cshogi import CSA
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('--diff', type=int, default=300)
parser.add_argument('--win_name')
args = parser.parse_args()

board = Board()
for filepath in glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True):
    for kif in CSA.Parser.parse_file(filepath):
        if kif.endgame not in ('%TORYO', '%KACHI'):
            continue
        if args.win_name and kif.names[kif.win - 1] != args.win_name:
            continue
        # •]‰¿’l‚ª‚È‚¢Šû•ˆ‚ğœŠO
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
                        i += 1
                    base_score = int(kif.scores[i] * 0.9)
                    ok = True
                    for after_score in kif.scores[i::2]:
                        if (base_score > 0 and after_score < base_score) or (base_score < 0 and after_score > base_score):
                            ok = False
                            break
                    if ok:
                        print(filepath, board.sfen(), sep='\t')
                        break

            board.push(move)
            prev_score = score
