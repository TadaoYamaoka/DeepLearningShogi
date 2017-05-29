from dlshogi.policy_network import *
from dlshogi.common import *

import cppshogi

from chainer import serializers

modelfile = r'H:\src\DeepLearningShogi\dlshogi\model_sl_elmo1000'
eval_dir = r'H:\src\elmo_for_learn\bin\20161007'

def main():
    while True:
        cmd_line = input()
        cmd = cmd_line.split(' ', 1)
        print('info string', cmd)

        if cmd[0] == 'usi':
            print('id name dlshogi')
            print('id author Tadao Yamaoka')
            print('usiok')
        elif cmd[0] == 'isready':
            # init cppshogi
            cppshogi.usi_init(eval_dir)

            model = PolicyNetwork()
            model.to_gpu()
            serializers.load_npz(modelfile, model)
            print('readyok')
        elif cmd[0] == 'usinewgame':
            continue
        elif cmd[0] == 'position':
            # cppshogi
            cppshogi.usi_position(cmd[1])
        elif cmd[0] == 'go':
            # cppshogi
            usi_move, usi_score = cppshogi.usi_go()
            #print(usi_move, usi_score)

            features1 = np.empty((1, 2 * 14, 9, 9), dtype=np.float32)
            features2 = np.empty((1, 2 * MAX_PIECES_IN_HAND_SUM + 1, 9, 9), dtype=np.float32)
            turn = cppshogi.usi_make_input_features(features1, features2)

            x1 = Variable(cuda.to_gpu(features1))
            x2 = Variable(cuda.to_gpu(features2))

            y = model(x1, x2, test=True)

            y_data = cuda.to_cpu(y.data)

            move = cppshogi.usi_select_move(y_data)
            
            # check score
            if usi_score >= 3000:
                print('bestmove', usi_move)
                continue
            elif usi_score < -3000:
                print('bestmove resign')
                continue

            print('bestmove', move)
        elif cmd[0] == 'quit':
            break

if __name__ == '__main__':
    main()