import argparse
from cshogi import *
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Make duplicate hcpe data unique')
parser.add_argument('hcpe', help='the input hcpe')
parser.add_argument('hcpe_uniq', help='the output hcpe')
parser.add_argument('--average', action='store_true', help='aggregate data with the same position and move, make the evaluation value the average value, and make the result the mode')
args = parser.parse_args()

hcpes = np.fromfile(args.hcpe, HuffmanCodedPosAndEval)
print(len(hcpes))

if args.average:
    df = pd.concat([pd.DataFrame(hcpes['hcp']), pd.DataFrame(hcpes[['eval', 'bestMove16', 'gameResult', 'dummy']])], axis=1)

    # hcpとbestMove16でグループ化して平均を算出
    df2 = df.groupby(list(range(32)) + ['bestMove16'], as_index=False).mean()

    # gameResultは最頻値に変換
    df2.loc[df2['gameResult'] >= 1.5, 'gameResult'] = 2
    df2.loc[(df2['gameResult'] > 0) & (df2['gameResult'] < 2), 'gameResult'] = 1

    hcpes2 = np.zeros(len(df2), HuffmanCodedPosAndEval)
    hcpes2['hcp'] = df2[list(range(32))]
    hcpes2['eval'] = df2['eval']
    hcpes2['bestMove16'] = df2['bestMove16']
    hcpes2['gameResult'] = df2['gameResult']
else:
    hcpes2 = np.unique(hcpes, axis=0)

hcpes2.tofile(args.hcpe_uniq)
print(len(hcpes2))
