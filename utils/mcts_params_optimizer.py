import numpy as np
import optuna
import subprocess
import argparse
import shogi
import random
from datetime import datetime
import os.path
import time
import math
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--command1', default=r'H:\src\DeepLearningShogi\x64\Release\usi.exe')
parser.add_argument('--command2', default=r'E:\game\shogi\ShogiGUI\gpsfish\gpsfish.exe')
parser.add_argument('--trials', type=int, default=100)
parser.add_argument('--model')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--hash_size', type=int)
parser.add_argument('--games', type=int, default=100)
parser.add_argument('--byoyomi', type=int, default=1000)
parser.add_argument('--resign', type=float, default=0.95)
parser.add_argument('--max_turn', type=int, default=256)
parser.add_argument('--initial_positions')
parser.add_argument('--kifu_dir')
parser.add_argument('--name')
parser.add_argument('--log', default=None)
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.INFO)

KIFU_TO_SQUARE_NAMES = [
    '９一', '８一', '７一', '６一', '５一', '４一', '３一', '２一', '１一',
    '９二', '８二', '７二', '６二', '５二', '４二', '３二', '２二', '１二',
    '９三', '８三', '７三', '６三', '５三', '４三', '３三', '２三', '１三',
    '９四', '８四', '７四', '６四', '５四', '４四', '３四', '２四', '１四',
    '９五', '８五', '７五', '６五', '５五', '４五', '３五', '２五', '１五',
    '９六', '８六', '７六', '６六', '５六', '４六', '３六', '２六', '１六',
    '９七', '８七', '７七', '６七', '５七', '４七', '３七', '２七', '１七',
    '９八', '８八', '７八', '６八', '５八', '４八', '３八', '２八', '１八',
    '９九', '８九', '７九', '６九', '５九', '４九', '３九', '２九', '１九',
]

KIFU_FROM_SQUARE_NAMES = [
    '91', '81', '71', '61', '51', '41', '31', '21', '11',
    '92', '82', '72', '62', '52', '42', '32', '22', '12',
    '93', '83', '73', '63', '53', '43', '33', '23', '13',
    '94', '84', '74', '64', '54', '44', '34', '24', '14',
    '95', '85', '75', '65', '55', '45', '35', '25', '15',
    '96', '86', '76', '66', '56', '46', '36', '26', '16',
    '97', '87', '77', '67', '57', '47', '37', '27', '17',
    '98', '88', '78', '68', '58', '48', '38', '28', '18',
    '99', '89', '79', '69', '59', '49', '39', '29', '19',
]

def sec_to_time(sec):
    h, m_ = divmod(math.ceil(sec), 60*60)
    m, s = divmod(m_, 60)
    return h, m, s

def kifu_header(kifu, starttime, names):
    kifu.write('開始日時：' + starttime.strftime('%Y/%m/%d %H:%M:%S\n'))
    kifu.write('手合割：平手\n')
    kifu.write('先手：' + names[0] + '\n')
    kifu.write('後手：' + names[1] + '\n')
    kifu.write('手数----指手---------消費時間--\n')

def kifu_move(board, move_usi):
    move = shogi.Move.from_usi(move_usi)
    move_to = KIFU_TO_SQUARE_NAMES[move.to_square]
    if board.move_number >= 2:
        prev_move = board.move_stack[-1]
        if prev_move.to_square == move.to_square:
            move_to = "同　"
    if move.from_square is not None:
        move_piece = shogi.PIECE_JAPANESE_SYMBOLS[board.piece_type_at(move.from_square)]
        if move.promotion:
            return '{}{}成({})'.format(
                move_to,
                move_piece,
                KIFU_FROM_SQUARE_NAMES[move.from_square],
                )
        else:
            return '{}{}({})'.format(
                move_to,
                move_piece,
                KIFU_FROM_SQUARE_NAMES[move.from_square],
                )
    else:
        move_piece = shogi.PIECE_JAPANESE_SYMBOLS[move.drop_piece_type]
        return '{}{}打'.format(
            move_to,
            move_piece
            )

def kifu_pv(board, items, i):
    if board.turn == shogi.BLACK:
        move_str = ' ▲'
    else:
        move_str = ' △'
    move = shogi.Move.from_usi(items[i])
    if move.promotion and board.piece_type_at(move.from_square) > shogi.KING:
        # 強制的に成った駒を移動する場合、PVを打ち切り
        return ''
    move_str += kifu_move(board, items[i])

    if i < len(items) - 1:
        board.push_usi(items[i])
        next_move = kifu_pv(board, items, i + 1)
        board.pop()
        return move_str + next_move
    else:
        return move_str

def kifu_line(kifu, board, move_usi, sec, sec_sum, info):

    m, s = divmod(math.ceil(sec), 60)
    h_sum, m_sum, s_sum = sec_to_time(sec_sum)

    if move_usi == 'resign':
        move_str = '投了        '
    elif move_usi == 'draw':
        move_str = '持将棋      '
    else:
        board.push_usi(move_usi)
        if board.is_fourfold_repetition():
            board.pop()
            move_str = '千日手      '
        else:
            board.pop()
            if move_usi[1:2] == '*':
                padding = '    '
            elif move_usi[-1] == '+':
                padding = ''
            else:
                padding = '  '
            move_str = kifu_move(board, move_usi) + padding

    kifu.write('{:>4} {}      ({:>2}:{:02}/{:02}:{:02}:{:02})\n'.format(
        board.move_number,
        move_str,
        m, s,
        h_sum, m_sum, s_sum))

    if info is not None:
        items = info.split(' ')
        comment = '**対局'
        i = 1
        while i < len(items):
            if items[i] == 'time':
                i += 1
                m, s = divmod(int(items[i]) / 1000, 60)
                s_str = '{:.1f}'.format(s)
                if s_str[1:2] == '.':
                    s_str = '0' + s_str
                comment += ' 時間 {:>02}:{}'.format(int(m), s_str)
            elif items[i] == 'depth':
                i += 1
                comment += ' 深さ {}'.format(items[i])
            elif items[i] == 'nodes':
                i += 1
                comment += ' ノード数 {}'.format(items[i])
            elif items[i] == 'score':
                i += 1
                if items[i] == 'cp':
                    i += 1
                    comment += ' 評価値 {}'.format(items[i] if board.turn == shogi.BLACK else -int(items[i]))
                elif items[i] == 'mate':
                    i += 1
                    if items[i][0:1] == '+':
                        comment += ' +詰' if board.turn == shogi.BLACK else ' -詰'
                    else:
                        comment += ' -詰' if board.turn == shogi.BLACK else ' +詰'
                    comment += str(items[i][1:])
            elif items[i] == 'pv':
                i += 1
                comment += kifu_pv(board, items, i)
            else:
                i += 1
        kifu.write(comment + '\n')

def objective(trial):
    C_init = trial.suggest_int('C_init', 50, 150)
    C_base = trial.suggest_int('C_base', 10000, 40000)

    logging.info('C_init = {}, C_base = {}'.format(C_init, C_base))

    win_count = 0
    draw_count = 0

    # 初期局面読み込み
    init_positions = []
    if args.initial_positions is not None:
        with open(args.initial_positions) as f:
            for line in f:
                init_positions.append(line.strip()[15:].split(' '))

    for n in range(args.games):
        logging.info('game {} start'.format(n))

        # 先後入れ替え
        if n % 2 == 0:
            command1 = args.command1
            command2 = args.command2
        else:
            command1 = args.command2
            command2 = args.command1

        # USIエンジン起動
        procs = [subprocess.Popen([command1], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(command1)),
                 subprocess.Popen([command2], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(command2))]

        names = []
        for i, p in enumerate(procs):
            logging.debug('pid = {}'.format(p.pid))
            if args.model is not None:
                p.stdin.write(b'setoption name DNN_Model value ' + args.model.encode('ascii') + b'\n')
            if args.batch_size is not None:
                p.stdin.write(b'setoption name DNN_Batch_Size value ' + str(args.batch_size).encode('ascii') + b'\n')
            if args.hash_size is not None:
                p.stdin.write(b'setoption name UCT_Hash value ' + str(args.hash_size).encode('ascii') + b'\n')
            p.stdin.write(b'setoption name Resign_Threshold value ' + str(args.resign).encode('ascii') + b'\n')
            p.stdin.write(b'setoption name USI_Ponder value false\n')

            # 最適化するパラメータ
            if n % 2 == i:
                p.stdin.write(b'setoption name C_init value ' + str(C_init).encode('ascii') + b'\n')
                p.stdin.write(b'setoption name C_base value ' + str(C_base).encode('ascii') + b'\n')

            p.stdin.write(b'usi\n')
            p.stdin.flush()

            while True:
                p.stdout.flush()
                line = p.stdout.readline()
                if line[:7] == b'id name':
                    names.append(line.strip()[8:].decode('ascii'))
                    logging.debug('name = {}'.format(names[-1]))
                elif line.strip() == b'usiok':
                    break

            p.stdin.write(b'isready\n')
            p.stdin.flush()

            while True:
                p.stdout.flush()
                line = p.stdout.readline()
                if line.strip() == b'readyok':
                    break

        if args.name is not None:
            names[0] = args.name

        # 棋譜ファイル初期化
        starttime = datetime.now()
        if args.kifu_dir is not None:
            kifu_dir = args.kifu_dir
        else:
            kifu_dir = ''
        kifu_path = os.path.join(kifu_dir, starttime.strftime('%Y%m%d_%H%M%S_') + names[0] + 'vs' + names[1] + '.kif')
        kifu = open(kifu_path, 'w')

        kifu_header(kifu, starttime, names)

        # 初期局面
        board = shogi.Board()
        if args.initial_positions is not None:
            if n % 2 == 0:
                init_position = random.choice(init_positions)
            for move_usi in init_position:
                kifu_line(kifu, board, move_usi, 0, 0, None)
                logging.info('{:>3} {}'.format(board.move_number, move_usi))
                board.push_usi(move_usi)


        # 新規ゲーム
        for p in procs:
            p.stdin.write(b'usinewgame\n')
            p.stdin.flush()

        # 対局
        is_game_over = False
        sec_sum = [0.0, 0.0]
        while not is_game_over:
            for i, p in enumerate(procs):
                info = None

                # 持将棋
                if board.move_number > args.max_turn:
                    kifu_line(kifu, board, 'draw', 0.0, sec_sum[i], None)
                    is_game_over = True
                    break

                # position
                line = 'position startpos moves'
                for m in board.move_stack:
                    line += ' ' + m.usi()
                p.stdin.write(line.encode('ascii') + b'\n')
                p.stdin.flush()

                # go
                line = 'go btime 0 wtime 0 byoyomi ' + str(args.byoyomi)
                p.stdin.write(line.encode('ascii') + b'\n')
                p.stdin.flush()
                time_start = time.time()

                is_resign = False
                while True:
                    p.stdout.flush()
                    line = p.stdout.readline().strip().decode('ascii')
                    logging.debug(line)
                    if line[:8] == 'bestmove':
                        sec = time.time() - time_start
                        sec_sum[i] += sec
                        move_usi = line[9:]
                        if info is not None:
                            logging.info(info)
                        logging.info('{:3} {}'.format(board.move_number, move_usi))
                        kifu_line(kifu, board, move_usi, sec, sec_sum[i], info)
                        # 詰みの場合、強制的に投了
                        if info is not None:
                            mate_p = info.find('mate ')
                            if mate_p > 0:
                                is_resign = True
                                if info[mate_p + 5] == '+':
                                    board.push_usi(move_usi)
                                break
                        if move_usi == 'resign':
                            is_resign = True
                        else:
                            board.push_usi(move_usi)
                        break
                    elif line[:4] == 'info' and line.find('pv ') > 0:
                        info = line

                # 終局判定
                if is_resign or board.is_game_over():
                    is_game_over = True
                    break

        # 棋譜に結果出力
        if not board.is_game_over() and board.move_number > args.max_turn:
            win = 2
            kifu.write('まで{}手で持将棋\n'.format(board.move_number))
        elif board.is_fourfold_repetition():
            win = 2
            kifu.write('まで{}手で千日手\n'.format(board.move_number - 2))
        else:
            win = shogi.BLACK if board.turn == shogi.WHITE else shogi.WHITE
            kifu.write('まで{}手で{}の勝ち\n'.format(board.move_number - 1, '先手' if win == shogi.BLACK else '後手'))
        kifu.close()

        # 勝敗カウント
        if n % 2 == 0 and win == shogi.BLACK or n % 2 == 1 and win == shogi.WHITE:
            win_count += 1
        elif win == 2:
            draw_count += 1

        if n + 1 == draw_count:
            win_rate = 0.0
        else:
            win_rate = win_count / (n + 1 - draw_count)

        logging.info('win : {}, win count = {}, draw count = {}, win rate = {:.1f}%'.format(
            win, win_count, draw_count, win_rate * 100))

        # USIエンジン終了
        for p in procs:
            p.stdin.write(b'quit\n')
            p.stdin.flush()
            p.wait()

        # 見込みのない最適化ステップを打ち切り
        trial.report(-win_rate, n)
        if trial.should_prune(n):
            raise optuna.structs.TrialPruned()

    # 勝率を負の値で返す
    return -win_rate

study = optuna.create_study()
study.optimize(objective, n_trials=args.trials)