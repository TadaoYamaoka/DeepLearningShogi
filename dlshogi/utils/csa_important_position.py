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
args = parser.parse_args()

board = Board()
lose_sfen = []
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
                        j = i + 1
                    else:
                        j = i
                    if kif.scores[j] > 100:
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
                                    # •‰‚¯‚½•û‚Ìè”Ô
                                    if (i + k) % 2 != j % 2:
                                        if (base_score > 0 and score > 100) or (base_score < 0 and score < -100):
                                            break
                                        lose_sfen.append('startpos moves ' + ' '.join([move_to_usi(move) for move in board.history]) + '\n')
                                    board.push(move)
                            break

            board.push(move)
            prev_score = score

if args.lose_sfen:
    with open(args.lose_sfen, 'w', newline='\n') as f:
        f.writelines(lose_sfen)
