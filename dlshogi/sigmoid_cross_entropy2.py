import numpy

from chainer import cuda
from chainer import function
from chainer.functions.activation import sigmoid
from chainer import utils
from chainer.utils import type_check

class SigmoidCrossEntropy2(function.Function):

    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.float32,
            x_type.shape == t_type.shape
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs

        # stable computation of the cross entropy.
        loss = t * xp.log1p(xp.exp(-x)) - (1 - t) * (xp.log(xp.exp(-x)) - xp.log1p(xp.exp(-x)))

        count = max(1, len(x))
        self.count = count

        return utils.force_array(
            xp.divide(xp.sum(loss), self.count, dtype=x.dtype)),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        gloss = grad_outputs[0]
        y, = sigmoid.Sigmoid().forward((x,))
        gx = xp.divide(
            gloss * (y - t), self.count,
            dtype=y.dtype)
        return gx, None


def sigmoid_cross_entropy2(x, t):
    return SigmoidCrossEntropy2()(x, t)
