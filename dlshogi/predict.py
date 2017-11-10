from chainer import cuda, Variable, serializers
import chainer.functions as F

from dlshogi.policy_value_network import *

model = PolicyValueNetwork()
model.to_gpu()

def load_model(file):
    print('Load model from', file)
    serializers.load_npz(file, model)

def predict(features1, features2):
    x1 = Variable(cuda.to_gpu(features1))
    x2 = Variable(cuda.to_gpu(features2))

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y1, y2 = model(x1, x2)

    return cuda.to_cpu(y1.data), cuda.to_cpu(F.tanh(y2).data)
