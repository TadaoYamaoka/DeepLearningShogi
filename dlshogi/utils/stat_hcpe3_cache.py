import argparse
from dlshogi.data_loader import Hcpe3DataLoader
from dlshogi.cppshogi import hcpe3_stat_cache

parser = argparse.ArgumentParser()
parser.add_argument('cache')
args = parser.parse_args()

data_len, actual_len = Hcpe3DataLoader.load_files([], cache=args.cache)
hcpe3_stat_cache()
