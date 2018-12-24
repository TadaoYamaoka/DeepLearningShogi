import os
import argparse
import random

parser = argparse.ArgumentParser(description='Prepare kifu list')
parser.add_argument('dir', type=str, help='directory')
parser.add_argument('filename', type=str, help='filename')
parser.add_argument('--ratio', '-r', type=float, default=0.9, help='Ratio of training data and test data')
args = parser.parse_args()

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

file_list = []
for filepath in find_all_files(args.dir):
    file_list.append(filepath)

shuffled = random.sample(file_list, len(file_list))
train_num = int(len(file_list) * args.ratio)

f_train = open(args.filename + '_train.txt', 'w')
f_test = open(args.filename + '_test.txt', 'w')

for path in shuffled[:train_num]:
    f_train.write(path + '\n')

for path in shuffled[train_num:]:
    f_test.write(path + '\n')

f_train.close()
f_test.close()