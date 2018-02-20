from chainer import cuda, Variable, serializers
import chainer.functions as F

from dlshogi.policy_value_network import *

model = []

def load_model(file, num_gpu=1):
    print('Load model from', file)
    for i in range(num_gpu):
        if len(model) <= i:
            model.append(PolicyValueNetwork())
        model[i].to_gpu(i)
        serializers.load_npz(file, model[i])

def predict(features1, features2, gpu=0):
    x1 = Variable(cuda.to_gpu(features1, gpu))
    x2 = Variable(cuda.to_gpu(features2, gpu))

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            y1, y2 = model[gpu](x1, x2)
            
            return cuda.to_cpu(y1.data), cuda.to_cpu(F.sigmoid(y2).data)
