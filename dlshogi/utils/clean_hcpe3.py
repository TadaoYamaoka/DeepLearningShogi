import argparse
from dlshogi import cppshogi

parser = argparse.ArgumentParser()
parser.add_argument('hcpe3')
parser.add_argument('cleaned')
args = parser.parse_args()

games, positions = cppshogi.hcpe3_clean(args.hcpe3, args.cleaned)

print('games', games)
print('positions', positions)
