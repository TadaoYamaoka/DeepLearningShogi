import optuna
from optuna import create_study, load_study
from optuna.pruners import MedianPruner
import argparse
import cshogi.cli
import sys
import logging

parser = argparse.ArgumentParser()
parser.add_argument('command1')
parser.add_argument('command2')
parser.add_argument('--options1', default='')
parser.add_argument('--options2', default='')
parser.add_argument('--study_name', default='mcts_params_optimizer')
parser.add_argument('--storage')
parser.add_argument('--trials', type=int, default=100)
parser.add_argument('--n_warmup_steps', type=int, default=30)
parser.add_argument('--games', type=int, default=100)
parser.add_argument('--byoyomi', type=int, default=1000)
parser.add_argument('--max_turn', type=int, default=320)
parser.add_argument('--opening')
parser.add_argument('--name')
parser.add_argument('--init_params', default='C_init:144,C_base:28288,C_fpu_reduction:27,C_init_root:116,C_base_root:25617,Softmax_Temperature:174')
parser.add_argument('--suggest_params', default='C_init:100~200,C_base:20000~50000,C_fpu_reduction:0~40,C_init_root:100~200,C_base_root:20000~50000,Softmax_Temperature:100~200')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

options_list = [{}, {}]
for i, kvs in enumerate([options.split(',') for options in (args.options1, args.options2)]):
    if len(kvs) == 1 and kvs[0] == '':
        continue
    for kv_str in kvs:
        kv = kv_str.split(':', 1)
        if len(kv) != 2:
            raise ValueError('options{} {}'.format(i + 1, kv_str))
        options_list[i][kv[0]] = kv[1]

def objective(trial):
    params = {}
    for k, v in [kv.split(':') for kv in args.suggest_params.split(',')]:
        params[k] = [int(x) for x in v.split('~')]

    if trial.number == 0 and args.init_params:
        for k, v in [kv.split(':') for kv in args.init_params.split(',')]:
            params[k] = [int(v)] * 2

    suggested = {}
    for k, v in params.items():
        suggested[k] = trial.suggest_int(k, v[0], v[1])

    print('Trial {} start. params = {}'.format(trial.number, str(suggested)))

    options1, options2 = options_list
    for k, v in suggested.items():
        options1[k] = v

    class Callback:
        def __init__(self):
            self.pruned = False

        def __call__(self, result):
            win_count = result['engine1_won']
            draw_count = result['draw']
            total_count = result['total']
            win_rate = (win_count + draw_count / 2) / total_count
            n = total_count - 1
            print('Trial {} game {} finished. win_count = {}, draw_count = {}, win_rate = {:.3f}'.format(trial.number, n, win_count, draw_count, win_rate))

            # 見込みのない最適化ステップを打ち切り
            trial.report(-win_rate, n)
            if trial.should_prune():
                self.pruned = True
                return False
            self.win_rate = win_rate
            return True

    # 先後入れ替えて対局
    callback = Callback()
    cshogi.cli.main(args.command1, args.command2, options1, options2, names=[args.name, None], games=args.games,
        mate_win=True, byoyomi=args.byoyomi, draw=args.max_turn, opening=args.opening, keep_process=True, is_display=False, debug=args.debug,
        callback=callback)

    if callback.pruned:
        raise optuna.TrialPruned()

    # 勝率を負の値で返す
    return -callback.win_rate

if args.storage:
    study = load_study(study_name=args.study_name, storage=args.storage, pruner=MedianPruner(n_warmup_steps=args.n_warmup_steps))
else:
    study = create_study(pruner=MedianPruner(n_warmup_steps=args.n_warmup_steps))
study.optimize(objective, n_trials=args.trials)
