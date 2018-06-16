from chainer import cuda, Variable, serializers

from dlshogi.policy_network import *
from dlshogi.common import *

model = PolicyNetwork()
model.to_gpu()

def load_model(file):
    print('Load model from', file)
    serializers.load_npz(file, model)

def predict(features1, features2):
    x1 = Variable(cuda.to_gpu(features1))
    x2 = Variable(cuda.to_gpu(features2))

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y = model(x1, x2)

    return cuda.to_cpu(y.data)

def dummy():
    pass

def dummy2(features1, features2):
    return np.zeros((1, 8181*81), dtype=np.float32)