import argparse
from cshogi import *
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()
parser.add_argument('hcpe')
parser.add_argument('fixed')
args = parser.parse_args()

# 評価値から勝率への変換
def eval_to_value(score, a):
    return 1.0 / (1.0 + np.exp(-score / a))

hcpes = np.fromfile(args.hcpe, HuffmanCodedPosAndEval)

turns = hcpes['hcp'][:,0] & 1 # hcpの1ビット目はturnを表す
signs = 1 - turns.astype(np.int8) * 2 # 後手の符号を反転
df = pd.DataFrame({'eval': hcpes['eval'] * signs, 'result': 2 - hcpes['gameResult']})

print(df['eval'].describe())
print(df['result'].describe())

# 詰みは除外する
df = df[df['eval'].abs() < 30000]
X = df['eval']
Y = df['result']

popt, pcov = curve_fit(eval_to_value, X, Y, p0=[600.0])
print(f'a = {popt[0]}')
print(f'stderr = {np.sqrt(pcov[0][0])}')

# dlshogiで使用している評価値と勝率変換の係数との比により、evalを補正する
hcpes['eval'] = np.clip(hcpes['eval'] * 756.0864962951762 / popt[0], a_min=-32767, a_max=32767).astype(np.int16)

hcpes.tofile(args.fixed)
