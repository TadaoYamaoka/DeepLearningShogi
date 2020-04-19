# 2つのbookをマージする
# 同一keyのcountの合計をbook1にそろえて足し合わせる

import argparse
import numpy as np
from collections import defaultdict
from cshogi import *

parser = argparse.ArgumentParser()
parser.add_argument('book1')
parser.add_argument('book2')
parser.add_argument('out')
parser.add_argument('--book2_ratio', type=float, default=0.5)
args = parser.parse_args()

book2_ratio = args.book2_ratio

book1 = np.fromfile(args.book1, BookEntry)
book2 = np.fromfile(args.book2, BookEntry)

book1dic = {}
for entry in book1:
    key = entry['key']
    if key not in book1dic:
        book1dic[key] = defaultdict(int)

    entries = book1dic[key]
    entries[entry['fromToPro']] += entry['count']

print(f"book1 entries   : {len(book1)}")
print(f"book1 positions : {len(book1dic)}")

book2dic = {}
for entry in book2:
    key = entry['key']
    if key not in book2dic:
        book2dic[key] = defaultdict(int)

    entries = book2dic[key]
    entries[entry['fromToPro']] += entry['count']

print(f"book2 entries   : {len(book2)}")
print(f"book2 positions : {len(book2dic)}")

for key, entries2 in book2dic.items():
    # book1にない場合、そのまま追加
    if key not in book1dic:
        book1dic[key] = entries2
        continue

    # book1にある場合、book1のcountの合計に合わせて追加する
    entries1 = book1dic[key]
    sum1 = 0
    for count in entries1.values():
        sum1 += count

    sum2 = 0
    for count in entries2.values():
        sum2 += count

    for fromToPro, count in entries2.items():
        entries1[fromToPro] = int(entries1[fromToPro] * (1 - book2_ratio) + count / sum1 * sum2 * book2_ratio)
        assert entries1[fromToPro] < 65536

num_entries = 0
for entries in book1dic.values():
    num_entries += len(entries)

# 保存
entries = np.empty(num_entries, dtype=BookEntry)
i = 0
for key in sorted(book1dic.keys()):
    for fromToPro, count in book1dic[key].items():
        entries[i] = key, fromToPro, count, 0
        i += 1
entries.tofile(args.out)

print(f"out entries   : {num_entries}")
print(f"out positions : {len(book1dic)}")
