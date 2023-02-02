import sys

moves = sys.argv[1].split(' ')
pos = 0

while True:
    cmd_line = input().strip()
    cmd = cmd_line.split(' ', 1)

    if cmd[0] == 'usi':
        print('id name repeater')
        print('usiok', flush=True)
    elif cmd[0] == 'isready':
        print('readyok', flush=True)
    elif cmd[0] == 'position':
        args = cmd[1].split(' moves ')
        if args[0] == 'startpos':
            pos = 0
            if len(args) > 1:
                for move in args[1].split():
                    if pos + 1 < len(moves) and moves[pos] == move:
                        pos += 1
                    else:
                        pos = -1
                        break
        else:
            pos = -1
    elif cmd[0] == 'go':
        print('bestmove ' + (moves[pos] if pos >= 0 else 'resign'), flush=True)
    elif cmd[0] == 'quit':
        break
