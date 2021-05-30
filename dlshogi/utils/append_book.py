from cshogi import *
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('book1')
parser.add_argument('book2')
parser.add_argument('book3')
args = parser.parse_args()

entry_num = 0
book3dic = {}
book2keys = set()
book2entries = np.fromfile(args.book2, dtype=BookEntry)
for entry in book2entries:
    key = entry['key']
    book2keys.add(key)
    if key not in book3dic:
        book3dic[key] = []
    book3dic[key].append(entry)
    entry_num += 1

# book1からbook2にないエントリを追加する
book1entries = np.fromfile(args.book1, dtype=BookEntry)
for entry in book1entries:
    key = entry['key']
    if key not in book2keys:
        if key not in book3dic:
            book3dic[key] = []
        book3dic[key].append(entry)
        entry_num += 1

print('book1 entry_num : {}'.format(len(book1entries)))
print('book2 entry_num : {}'.format(len(book2entries)))
print('book3 entry_num : {}'.format(entry_num))

# 保存
book_entries = np.empty(entry_num, dtype=BookEntry)
i = 0
for key in sorted(book3dic.keys()):
    entries = book3dic[key]
    for entry in entries:
        book_entries[i] = entry
        i += 1
book_entries.tofile(args.book3)
