def usi(player):
    while True:
        cmd_line = input()
        cmd = cmd_line.split(' ', 1)

        if cmd[0] == 'usi':
            player.usi()
        elif cmd[0] == 'setoption':
            option = cmd[1].split(' ')
            player.setoption(option)
        elif cmd[0] == 'isready':
            player.isready()
        elif cmd[0] == 'usinewgame':
            player.usinewgame()
        elif cmd[0] == 'position':
            moves = cmd[1].split(' ')
            player.position(moves)
        elif cmd[0] == 'go':
            player.go()
        elif cmd[0] == 'quit':
            player.quit()
            break
