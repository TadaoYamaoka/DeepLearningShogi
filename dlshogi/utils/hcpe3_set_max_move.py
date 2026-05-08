import argparse
import os
import tempfile


HCPE3_HEADER_SIZE = 36
HCPE3_MOVE_INFO_SIZE = 6
HCPE3_MOVE_VISITS_SIZE = 4
HCPE3_MOVE_NUM_OFFSET = 32
HCPE3_GAME_INFO_OFFSET = 35
HCPE3_MAX_MOVE_MASK = 0x0c
HCPE3_MAX_MOVE_SHIFT = 2
HCPE3_MAX_MOVE_CODES = {
    256: 1,
    320: 2,
    512: 3,
}
HCPE3_MAX_MOVE_NUM = 513
HCPE3_MAX_CANDIDATE_NUM = 593


def patch_hcpe3_max_move(src, dst, max_move_code):
    games = 0
    positions = 0
    old_code_counts = [0, 0, 0, 0]

    while True:
        header = bytearray(src.read(HCPE3_HEADER_SIZE))
        if len(header) == 0:
            break
        if len(header) != HCPE3_HEADER_SIZE:
            raise RuntimeError(f'truncated hcpe3 header at game {games}')

        move_num = int.from_bytes(
            header[HCPE3_MOVE_NUM_OFFSET:HCPE3_MOVE_NUM_OFFSET + 2],
            'little',
        )
        if move_num > HCPE3_MAX_MOVE_NUM:
            raise RuntimeError(f'invalid moveNum {move_num} at game {games}')

        old_game_info = header[HCPE3_GAME_INFO_OFFSET]
        old_code = (old_game_info & HCPE3_MAX_MOVE_MASK) >> HCPE3_MAX_MOVE_SHIFT
        old_code_counts[old_code] += 1
        header[HCPE3_GAME_INFO_OFFSET] = (
            old_game_info & ~HCPE3_MAX_MOVE_MASK
        ) | (max_move_code << HCPE3_MAX_MOVE_SHIFT)
        dst.write(header)

        for i in range(move_num):
            move_info = src.read(HCPE3_MOVE_INFO_SIZE)
            if len(move_info) != HCPE3_MOVE_INFO_SIZE:
                raise RuntimeError(f'truncated MoveInfo at game {games}, ply {i}')
            dst.write(move_info)

            candidate_num = int.from_bytes(move_info[4:6], 'little')
            if candidate_num > HCPE3_MAX_CANDIDATE_NUM:
                raise RuntimeError(f'invalid candidateNum {candidate_num} at game {games}, ply {i}')

            visits_size = HCPE3_MOVE_VISITS_SIZE * candidate_num
            move_visits = src.read(visits_size)
            if len(move_visits) != visits_size:
                raise RuntimeError(f'truncated MoveVisits at game {games}, ply {i}')
            dst.write(move_visits)

            if candidate_num > 0:
                positions += 1

        games += 1

    return games, positions, old_code_counts


def output_path(args):
    if args.output:
        return args.output
    return args.hcpe3


def main():
    parser = argparse.ArgumentParser(
        description='Overwrite the HCPE3 gameInfo max-move code while preserving opponent bits.',
    )
    parser.add_argument('hcpe3', help='input hcpe3 file')
    parser.add_argument('max_move', type=int, choices=HCPE3_MAX_MOVE_CODES.keys(), help='maximum move number to store')
    parser.add_argument('-o', '--output', help='output hcpe3 file; omitted means update input file in place')
    args = parser.parse_args()

    max_move_code = HCPE3_MAX_MOVE_CODES[args.max_move]
    out_path = output_path(args)
    inplace = os.path.abspath(out_path) == os.path.abspath(args.hcpe3)

    if inplace:
        out_dir = os.path.dirname(os.path.abspath(args.hcpe3)) or '.'
        fd, tmp_path = tempfile.mkstemp(prefix='.set_hcpe3_max_move.', suffix='.tmp', dir=out_dir)
        os.close(fd)
        try:
            with open(args.hcpe3, 'rb') as src, open(tmp_path, 'wb') as dst:
                games, positions, old_code_counts = patch_hcpe3_max_move(src, dst, max_move_code)
            os.replace(tmp_path, args.hcpe3)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    else:
        with open(args.hcpe3, 'rb') as src, open(out_path, 'wb') as dst:
            games, positions, old_code_counts = patch_hcpe3_max_move(src, dst, max_move_code)

    print('games', games)
    print('positions', positions)
    print('old max_move code counts', old_code_counts)
    print('max_move', args.max_move)
    print('output', out_path)


if __name__ == '__main__':
    main()