import argparse
from cshogi import HuffmanCodedPosAndEval, BLACK_WIN, DRAW
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("hcpe")
parser.add_argument("--csv")
args = parser.parse_args()

hcpes = np.fromfile(args.hcpe, HuffmanCodedPosAndEval)

print("positions", len(hcpes))

color = hcpes["hcp"][:, 0] & 1
evals = hcpes["eval"]
result = hcpes["gameResult"]

black_win = (result == BLACK_WIN).astype(np.uint8)
draw = (result == DRAW).astype(np.uint8)

df = pd.DataFrame(
    {
        "color": color,
        "black win": black_win,
        "draw": draw,
        "eval": evals,
    }
)

if args.csv:
    df.to_csv(args.csv)

print(df.describe())
print("black win rate", df["black win"].sum() / (len(df) - df["draw"].sum()))
print(
    "white win rate",
    (len(df) - df["black win"].sum() - df["draw"].sum()) / (len(df) - df["draw"].sum()),
)

print("\nunique positions:")
_, hcp_counts = np.unique(hcpes["hcp"], axis=0, return_counts=True)
print(pd.DataFrame(hcp_counts, columns=["hcp"]).describe())

print("\nunique positions and best moves:")
hcp_move = hcpes[["hcp", "bestMove16"]]
_, hcp_move_counts = np.unique(hcp_move, axis=0, return_counts=True)
print(pd.DataFrame(hcp_move_counts, columns=["(hcp, bestMove16)"]).describe())
