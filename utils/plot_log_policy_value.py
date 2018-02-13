import argparse
import re
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('log', type=str)
args = parser.parse_args()

ptn = re.compile(r'iteration = ([0-9]+), loss = [0-9.]+, [0-9.]+, [0-9.]+, ([0-9.]+), .* test accuracy = ([0-9.]+), ([0-9.]+)')

iteration_list = []
loss_list = []
accuracy1_list = []
accuracy2_list = []
for line in open(args.log, 'r'):
    m = ptn.search(line)
    if m:
        iteration_list.append(int(m.group(1)))
        loss_list.append(float(m.group(2)))
        accuracy1_list.append(float(m.group(3)))
        accuracy2_list.append(float(m.group(4)))

fig, ax1 = plt.subplots()
p1, = ax1.plot(iteration_list, loss_list, 'b', label='train loss')
ax1.set_xlabel('iterations')

ax2=ax1.twinx()
p2, = ax2.plot(iteration_list, accuracy1_list, 'g', label='policy accuracy')
p3, = ax2.plot(iteration_list, accuracy2_list, 'r', label='value accuracy')

ax1.legend(handles=[p1, p2, p3])

plt.show()
