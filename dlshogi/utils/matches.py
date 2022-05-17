import subprocess
import argparse
import random
from datetime import datetime
from collections import defaultdict
import os.path
import time
import math
import logging

import cshogi
from cshogi import KIF

parser = argparse.ArgumentParser()
parser.add_argument('command1')
parser.add_argument('command2')
parser.add_argument('--options1', default='')
parser.add_argument('--options2', default='')
parser.add_argument('--games', type=int, default=100)
parser.add_argument('--byoyomi', type=int, default=1000)
parser.add_argument('--max_turn', type=int, default=256)
parser.add_argument('--initial_positions')
parser.add_argument('--kifu_dir')
parser.add_argument('--log', default=None)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG if args.debug else logging.INFO)

options_list = [{}, {}]
for i, kvs in enumerate([options.split(',') for options in (args.options1, args.options2)]):
    if len(kvs) == 1 and kvs[0] == '':
        continue
    for kv_str in kvs:
        kv = kv_str.split(':', 1)
        if len(kv) != 2:
            raise ValueError('options{} {}'.format(i + 1, kv_str))
        options_list[i][kv[0]] = kv[1]

def main():
    win_count = 0
    draw_count = 0

    # 初期局面読み込み
    init_positions = []
    if args.initial_positions is not None:
        with open(args.initial_positions) as f:
            for line in f:
                init_positions.append(line.strip()[15:].split(' '))

    for n in range(args.games):
        if __debug__: logging.debug('game {} start'.format(n))

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
            if __debug__: logging.debug('pid = {}'.format(p.pid))
            p.stdin.write(b'setoption name USI_Ponder value false\n')
            for name, value in options_list[(n + i) % 2].items():
                if __debug__: logging.debug('usi option {} {}'.format(name, value))
                p.stdin.write('setoption name {} value {}\n'.format(name, value).encode('ascii'))

            p.stdin.write(b'usi\n')
            p.stdin.flush()

            while True:
                p.stdout.flush()
                line = p.stdout.readline()
                if line[:7] == b'id name':
                    names.append(line.strip()[8:].decode('ascii'))
                    if __debug__: logging.debug('name = {}'.format(names[-1]))
                elif line.strip() == b'usiok':
                    break

            p.stdin.write(b'isready\n')
            p.stdin.flush()

            while True:
                p.stdout.flush()
                line = p.stdout.readline()
                if line.strip() == b'readyok':
                    break

        # 棋譜ファイル初期化
        starttime = datetime.now()
        if args.kifu_dir is not None:
            kifu_path = os.path.join(args.kifu_dir, starttime.strftime('%Y%m%d_%H%M%S_') + names[0] + 'vs' + names[1] + '.kif')
            kifu = KIF.Exporter(kifu_path)

            kifu.header(names, starttime)

        # 初期局面
        board = cshogi.Board()
        if args.initial_positions is not None:
            if n % 2 == 0:
                init_position = random.choice(init_positions)
            for move_usi in init_position:
                move = board.move_from_usi(move_usi)
                if args.kifu_dir is not None: kifu.move(move)
                if __debug__: logging.debug('{:>3} {}'.format(board.move_number, move_usi))
                board.push(move)


        # 新規ゲーム
        for p in procs:
            p.stdin.write(b'usinewgame\n')
            p.stdin.flush()

        # 対局
        is_game_over = False
        sec_sum = [0.0, 0.0]
        position = 'position startpos moves'
        repetition_hash = defaultdict(int)
        while not is_game_over:
            for i, p in enumerate(procs):
                if __debug__: engine_id = 'engine1' if (n + i) % 2 == 0 else 'engine2'
                info = None

                # 持将棋
                if board.move_number > args.max_turn:
                    is_game_over = True
                    break

                # position
                line = position
                if __debug__: logging.debug('[{}] {}'.format(engine_id, line))
                p.stdin.write(line.encode('ascii') + b'\n')
                p.stdin.flush()

                # go
                line = 'go btime 0 wtime 0 byoyomi ' + str(args.byoyomi)
                p.stdin.write(line.encode('ascii') + b'\n')
                p.stdin.flush()
                time_start = time.time()

                is_resign = False
                is_nyugyoku = False
                while True:
                    p.stdout.flush()
                    line = p.stdout.readline().strip().decode('ascii')
                    if __debug__: logging.debug('[{}] {}'.format(engine_id, line))
                    if line[:8] == 'bestmove':
                        sec = time.time() - time_start
                        sec_sum[i] += sec
                        move_usi = line[9:].split(' ', 1)[0]
                        if args.kifu_dir is not None:
                            kifu.move(board.move_from_usi(move_usi), sec, sec_sum[i])
                            if info is not None:
                                kifu.info(info)
                        # 詰みの場合、強制的に投了
                        if info is not None:
                            mate_p = info.find('mate ')
                            if mate_p > 0:
                                is_resign = True
                                if info[mate_p + 5] != '-':
                                    board.push_usi(move_usi)
                                break
                        if move_usi == 'resign':
                            # 投了
                            is_resign = True
                        elif move_usi == 'win':
                            # 入玉勝ち宣言
                            is_nyugyoku = True
                        else:
                            board.push_usi(move_usi)
                            position += ' ' + move_usi
                            repetition_hash[board.zobrist_hash()] += 1
                        break
                    elif line[:4] == 'info' and line.find('score ') > 0:
                        info = line

                # 終局判定
                repetition = board.is_draw()
                if repetition in [cshogi.REPETITION_DRAW, cshogi.REPETITION_WIN, cshogi.REPETITION_LOSE] and repetition_hash[board.zobrist_hash()] == 4:
                    break
                repetition = cshogi.NOT_REPETITION
                if is_resign or is_nyugyoku or board.is_game_over():
                    is_game_over = True
                    break

        # 棋譜に結果出力
        if repetition == REPETITION_DRAW:
            win = 2
            if args.kifu_dir is not None:
                kifu.end('sennichite', 0.0, sec_sum[i])
        elif repetition == REPETITION_WIN:
            win = cshogi.BLACK if board.turn == cshogi.WHITE else cshogi.WHITE
            if args.kifu_dir is not None:
                kifu.end('illegal_win', 0.0, sec_sum[i])
        elif repetition == REPETITION_LOSE:
            win = cshogi.WHITE if board.turn == cshogi.WHITE else cshogi.BLACK
            if args.kifu_dir is not None:
                kifu.end('illegal_lose', 0.0, sec_sum[i])
        elif not board.is_game_over() and board.move_number > args.max_turn:
            win = 2
            if args.kifu_dir is not None:
                kifu.end('draw', 0.0, sec_sum[i])
        elif is_nyugyoku:
            win = board.turn
            if args.kifu_dir is not None:
                kifu.end('win', sec, sec_sum[i])
        else:
            win = cshogi.BLACK if board.turn == cshogi.WHITE else cshogi.WHITE
            if args.kifu_dir is not None:
                if is_resign:
                    kifu.end('resign', sec, sec_sum[i])
                else:
                    kifu.end('resign', 0.0, sec_sum[i])
        if args.kifu_dir is not None:
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

        logging.info('game {} result : win = {}, win count = {}, draw count = {}, win rate = {:.1f}%'.format(
            n, win, win_count, draw_count, win_rate * 100))

        # USIエンジン終了
        for p in procs:
            p.stdin.write(b'quit\n')
            p.stdin.flush()
            p.wait()

main()
