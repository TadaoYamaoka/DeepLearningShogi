import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('pgn', type=str, nargs='+')
parser.add_argument('outprefix')
parser.add_argument('--uniq', action='store_true')
args = parser.parse_args()

pgns = defaultdict(list)
stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0, 0]))

def append(players, pgntext, result):
    key = '+'.join(sorted(players))
    if not args.uniq or pgntext not in pgns[key]:
        pgns[key].append(pgntext)
        stat = stats[key]
        if result == '1-0':
            stat[players[0]][0] += 1
            stat[players[1]][3] += 1
        elif result == '0-1':
            stat[players[0]][2] += 1
            stat[players[1]][1] += 1
        else:
            stat[players[0]][4] += 1
            stat[players[1]][5] += 1

for file in args.pgn:
    header = False
    players = []
    pgntext = ""
    for line in open(file):
        if line[:1] == "[":
            if not header and pgntext != "":
                append(players, pgntext, result)
                players = []
                pgntext = ""

            header = True

            if line[1:6] in ['White', 'Black']:
                players.append(line[8:-3])
            elif line[1:7] == 'Result':
                result = line[9:-3]
        else:
            header = False

        pgntext += line

    append(players, pgntext, result)

for key, pgntext in pgns.items():
    with open(args.outprefix + key + '.pgn', 'w') as f:
        f.writelines(pgntext)

    print()
    players = key.split('+')
    stat = stats[key]
    engine1_won = stat[players[0]]
    engine2_won = stat[players[1]]
    black_won = engine1_won[0] + engine2_won[0]
    white_won = engine1_won[1] + engine2_won[1]
    engine1_won_sum = engine1_won[0] + engine1_won[1]
    engine2_won_sum = engine2_won[0] + engine2_won[1]
    draw_count = engine1_won[4] + engine1_won[5]
    total_count = engine1_won_sum + engine2_won_sum + draw_count
    print('{} vs {}: {}-{}-{} ({:.1f}%)'.format(
        players[0], players[1], engine1_won_sum, engine2_won_sum, draw_count,
        (engine1_won_sum + draw_count / 2) / total_count * 100))
    print('Black vs White: {}-{}-{} ({:.1f}%)'.format(
        black_won, white_won, draw_count,
        (black_won + draw_count / 2) / total_count * 100))
    print('{} playing Black: {}-{}-{} ({:.1f}%)'.format(
        players[0],
        engine1_won[0], engine1_won[2], engine1_won[4],
        (engine1_won[0] + engine1_won[4] / 2) / (engine1_won[0] + engine1_won[2] + engine1_won[4]) * 100))
    print('{} playing White: {}-{}-{} ({:.1f}%)'.format(
        players[0],
        engine1_won[1], engine1_won[3], engine1_won[5],
        (engine1_won[1] + engine1_won[5] / 2) / (engine1_won[1] + engine1_won[3] + engine1_won[5]) * 100))
    print('{} playing Black: {}-{}-{} ({:.1f}%)'.format(
        players[1],
        engine2_won[0], engine2_won[2], engine2_won[4],
        (engine2_won[0] + engine2_won[4] / 2) / (engine2_won[0] + engine2_won[2] + engine2_won[4]) * 100))
    print('{} playing White: {}-{}-{} ({:.1f}%)'.format(
        players[1],
        engine2_won[1], engine2_won[3], engine2_won[5],
        (engine2_won[1] + engine2_won[5] / 2) / (engine2_won[1] + engine2_won[3] + engine2_won[5]) * 100))
