import argparse
import re
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('logs', type=str, nargs='+')
parser.add_argument('--testloss', action='store_true')
parser.add_argument('--grid', action='store_true')
args = parser.parse_args()

traintest = 'test' if args.testloss else 'train'
ptn = re.compile(r'steps = ([0-9]+),.* '+ traintest + r' loss = [0-9.]+, [0-9.]+, [0-9.]+, ([0-9.]+),.* test accuracy = ([0-9.]+), ([0-9.]+)')

step_list = []
loss_list = []
accuracy1_list = []
accuracy2_list = []
for log in args.logs:
    for line in open(log, 'r'):
        m = ptn.search(line)
        if m:
            step_list.append(int(m.group(1)))
            loss_list.append(float(m.group(2)))
            accuracy1_list.append(float(m.group(3)))
            accuracy2_list.append(float(m.group(4)))

fig, ax1 = plt.subplots()
p1, = ax1.plot(step_list, loss_list, 'b', label=traintest + ' loss')
ax1.set_xlabel('steps')
ax1.set_ylabel('loss')

ax2=ax1.twinx()
p2, = ax2.plot(step_list, accuracy1_list, 'g', label='policy accuracy')
p3, = ax2.plot(step_list, accuracy2_list, 'r', label='value accuracy')
ax2.set_ylabel('accuracy')

ax1.legend(handles=[p1, p2, p3])
if args.grid:
    ax1.grid()

plt.show()
