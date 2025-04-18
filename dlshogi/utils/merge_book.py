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
parser.add_argument('--book2_ratio', type=float)
args = parser.parse_args()

book2_ratio = args.book2_ratio

book1 = np.fromfile(args.book1, BookEntry)
book2 = np.fromfile(args.book2, BookEntry)

book1dic = {}
for entry in book1:
    key = entry['key']
    if key not in book1dic:
        book1dic[key] = defaultdict(lambda: [0, None])

    entries = book1dic[key]
    entries[entry['fromToPro']] = [entry['count'], entry['score']]

print(f"book1 entries   : {len(book1)}")
print(f"book1 positions : {len(book1dic)}")

book2dic = {}
for entry in book2:
    key = entry['key']
    if key not in book2dic:
        book2dic[key] = defaultdict(lambda: [0, 0])

    entries = book2dic[key]
    entries[entry['fromToPro']] = [entry['count'], entry['score']]

print(f"book2 entries   : {len(book2)}")
print(f"book2 positions : {len(book2dic)}")

for key, entries2 in book2dic.items():
    # book1にない場合、そのまま追加
    if key not in book1dic:
        book1dic[key] = entries2
        continue

    if args.book2_ratio:
        # book1にある場合、book1のcountの合計に合わせて追加する
        entries1 = book1dic[key]
        sum1 = 0
        for (count, score) in entries1.values():
            sum1 += count

        sum2 = 0
        for (count, score) in entries2.values():
            sum2 += count

        sum3 = min(sum1 + sum2, 65535)

        for fromToPro, (count, score) in entries1.items():
            entries1[fromToPro] = [int(count / sum1 * (1 - book2_ratio) * sum3), score]

        for fromToPro, (count, score) in entries2.items():
            values = entries1[fromToPro]
            values[0] += int(count / sum2 * book2_ratio * sum3)
            values[1] = score if values[1] is None else int(values[1] * (1 - book2_ratio) + score * book2_ratio)
            assert entries1[fromToPro][0] < 65536
    else:
        # book2で上書きする
        book1dic[key] =entries2

num_entries = 0
for entries in book1dic.values():
    num_entries += len(entries)

# 保存
entries = np.empty(num_entries, dtype=BookEntry)
i = 0
for key in sorted(book1dic.keys()):
    for fromToPro, (count, score) in book1dic[key].items():
        entries[i] = key, fromToPro, count, score
        i += 1
entries.tofile(args.out)

print(f"out entries   : {num_entries}")
print(f"out positions : {len(book1dic)}")
