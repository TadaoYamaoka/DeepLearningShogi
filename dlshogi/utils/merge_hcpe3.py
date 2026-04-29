import argparse
from dlshogi import cppshogi


parser = argparse.ArgumentParser(description='Merge hcpe3 files into one hcpe3 file.')
parser.add_argument('hcpe3', nargs='+', help='input hcpe3 files')
parser.add_argument('-o', '--output', required=True, help='output hcpe3 file')
args = parser.parse_args()

cppshogi.hcpe3_merge(args.hcpe3, args.output)
