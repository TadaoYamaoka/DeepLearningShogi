from chainer import cuda, Variable, serializers

from dlshogi.policy_network import *
from dlshogi.common import *

import cppshogi

import argparse
import time
import threading

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--thread', '-t', type=int, default=1, help='Number of thread')
parser.add_argument('--loop_count', '-l', type=int, default=1, help='Number of loop count')
parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
args = parser.parse_args()

model = PolicyNetwork()
model.to_gpu()

print('Load model from', args.model)
serializers.load_npz(args.model, model)

print('thread:', args.thread)
print('loop count:', args.loop_count)
print('batch size:', args.batch_size)

engines = [cppshogi.Engine() for _ in range(args.thread)]

def run(engine):
    engine.position('startpos')
    features1 = np.empty((args.batch_size, 2 * 14, 9, 9), dtype=np.float32)
    features2 = np.empty((args.batch_size, 2 * MAX_PIECES_IN_HAND_SUM + 1, 9, 9), dtype=np.float32)
    [engine.make_input_features(features1[i], features2[i]) for i in range(args.batch_size)]
    for i in range(args.loop_count):
        x1 = Variable(cuda.to_gpu(features1))
        x2 = Variable(cuda.to_gpu(features2))

        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                y = model(x1, x2)

        y_data = cuda.to_cpu(y.data)

        #print(y_data)
        #move = engine.select_move(y_data)
        #print(move)

th = [threading.Thread(target=run, args=([engines[i]])) for i in range(args.thread)]

start = time.time()

for i in range(args.thread):
    th[i].start()

for i in range(args.thread):
    th[i].join()

elapsed_time = time.time() - start
print('elapsed_time:', elapsed_time, '[sec]')

print('throughput:', args.thread * args.loop_count * args.batch_size / elapsed_time, '[nps]')