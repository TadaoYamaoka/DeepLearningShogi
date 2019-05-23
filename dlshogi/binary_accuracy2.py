from __future__ import division

from chainer import backend
from chainer import function
from chainer.utils import type_check


class BinaryAccuracy2(function.Function):

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x', 't'))
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype.kind == 'f',
            t_type.shape == x_type.shape,
        )

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        y, t = inputs
        # flatten
        y = y.ravel()
        t = t.ravel()
        c = (y > 0)
        count = len(t)
        return xp.asarray((c == t).sum() / count, dtype=y.dtype),


def binary_accuracy2(y, t):
    return BinaryAccuracy2()(y, t)