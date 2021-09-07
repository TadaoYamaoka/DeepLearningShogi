import argparse
import sys
import re
import glob
import pandas as pd

def plot_log_policy_value(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', type=str, nargs='+')
    parser.add_argument('--testloss', action='store_true')
    parser.add_argument('--grid', action='store_true')
    args = parser.parse_args(argv)

    traintest = 'test' if args.testloss else 'train'
    ptn = re.compile(r'steps = ([0-9]+),.* '+ traintest + r' loss = [0-9.]+, [0-9.]+, [0-9.]+, ([0-9.]+),.* test accuracy = ([0-9.]+), ([0-9.]+)')

    logs = []
    for log in args.logs:
        logs.extend(glob.glob(log))

    step_list = []
    loss_list = []
    accuracy1_list = []
    accuracy2_list = []
    for log in logs:
        for line in open(log, 'r'):
            m = ptn.search(line)
            if m:
                step_list.append(int(m.group(1)))
                loss_list.append(float(m.group(2)))
                accuracy1_list.append(float(m.group(3)))
                accuracy2_list.append(float(m.group(4)))

    df = pd.DataFrame({
        'steps': step_list,
        traintest + ' loss': loss_list,
        'policy accuracy': accuracy1_list,
        'value accuracy': accuracy2_list}).set_index('steps').sort_index()
    ax = df.plot(secondary_y=['policy accuracy', 'value accuracy'], grid=args.grid)
    ax.set_ylabel('loss')
    ax.right_ax.set_ylabel('accuracy')

if __name__ == '__main__':
    plot_log_policy_value(*sys.argv[1:])
