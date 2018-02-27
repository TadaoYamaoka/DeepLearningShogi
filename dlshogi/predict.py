from chainer import cuda, Variable, serializers
import chainer.functions as F

from dlshogi.policy_value_network import *

model = {}

def load_model(file, gpu=0):
    if not model.get(gpu):
        model[gpu] = PolicyValueNetwork()
        model[gpu].to_gpu(gpu)
    print('compute_capability ', chainer.cuda.cuda.Device(gpu).compute_capability)
    print('Load model from', file)
    serializers.load_npz(file, model[gpu])

def predict(features1, features2, gpu=0):
    x1 = Variable(cuda.to_gpu(features1, gpu))
    x2 = Variable(cuda.to_gpu(features2, gpu))

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y1, y2 = model[gpu](x1, x2)

            return cuda.to_cpu(y1.data), cuda.to_cpu(F.sigmoid(y2).data)
