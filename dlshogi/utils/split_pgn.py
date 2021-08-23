import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('pgn', type=str, nargs='+')
parser.add_argument('outprefix')
args = parser.parse_args()

pgns = defaultdict(str)

for file in args.pgn:
    header = False
    players = []
    pgntext = ""
    for line in open(file):
        if line[:1] == "[":
            if not header and pgntext != "":
                key = '+'.join(sorted(players))
                pgns[key] += pgntext
                players = []
                pgntext = ""

            header = True

            if line[1:6] in ["White", "Black"]:
                players.append(line[8:-3])
        else:
            header = False

        pgntext += line

    key = '+'.join(sorted(players))
    pgns[key] += pgntext

for key, pgntext in pgns.items():
    with open(args.outprefix + key + '.pgn', 'w') as f:
        f.write(pgntext)
