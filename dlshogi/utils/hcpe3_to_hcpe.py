import argparse
from dlshogi import cppshogi

parser = argparse.ArgumentParser()
parser.add_argument('hcpe3')
parser.add_argument('hcpe')
args = parser.parse_args()

games, positions = cppshogi.hcpe3_to_hcpe(args.hcpe3, args.hcpe)

print('games', games)
print('positions', positions)

