import numpy as np
import chainer
from chainer import serializers
import chainer.links as L
from dlshogi.policy_value_network import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('fused_model')
args = parser.parse_args()

def fuse_conv_and_bn(conv, bn):
    # init
    fusedconv = L.Convolution2D(
        conv.W.shape[1],
        conv.out_channels,
        ksize=conv.ksize,
        stride=conv.stride,
        pad=conv.pad
    )

    # prepare filters
    w_conv = conv.W.data.reshape(conv.out_channels, -1)
    w_bn = np.diag(np.divide(bn.gamma.data, np.sqrt(bn.eps + bn.avg_var)))
    np.copyto(fusedconv.W.data, np.matmul(w_bn, w_conv).reshape(fusedconv.W.data.shape))

    # prepare spatial bias
    if conv.b is not None:
        b_conv = conv.b.data
    else:
        b_conv = np.zeros(conv.W.data.shape[0])
    b_bn = bn.beta.data - np.divide(np.multiply(bn.gamma.data, bn.avg_mean), np.sqrt(bn.avg_var + bn.eps))
    np.copyto(fusedconv.b.data, b_conv + b_bn)

    return fusedconv

class FusedPolicyValueNetwork(Chain):
    def __init__(self, model):
        super(FusedPolicyValueNetwork, self).__init__(
            l1_1_1=model.l1_1_1,
            l1_1_2=model.l1_1_2,
            l1_2=model.l1_2,
            l2=fuse_conv_and_bn(model.l2, model.norm2),
            l3=model.l3,
            l4=fuse_conv_and_bn(model.l4, model.norm4),
            l5=model.l5,
            l6=fuse_conv_and_bn(model.l6, model.norm6),
            l7=model.l7,
            l8=fuse_conv_and_bn(model.l8, model.norm8),
            l9=model.l9,
            l10=fuse_conv_and_bn(model.l10, model.norm10),
            l11=model.l11,
            l12=fuse_conv_and_bn(model.l12, model.norm12),
            l13=model.l13,
            l14=fuse_conv_and_bn(model.l14, model.norm14),
            l15=model.l15,
            l16=fuse_conv_and_bn(model.l16, model.norm16),
            l17=model.l17,
            l18=fuse_conv_and_bn(model.l18, model.norm18),
            l19=model.l19,
            l20=fuse_conv_and_bn(model.l20, model.norm20),
            l21=model.l21,
            # policy network
            l22=model.l22,
            l22_2=model.l22_2,
            # value network
            l22_v=model.l22_v,
            l23_v=model.l23_v,
            l24_v=model.l24_v,
            norm1=model.norm1,
            norm3=model.norm3,
            norm5=model.norm5,
            norm7=model.norm7,
            norm9=model.norm9,
            norm11=model.norm11,
            norm13=model.norm13,
            norm15=model.norm15,
            norm17=model.norm17,
            norm19=model.norm19,
            norm21=model.norm21,
            norm22_v=model.norm22_v
        )

model = PolicyValueNetwork()
serializers.load_npz(args.model, model)

fused_model= FusedPolicyValueNetwork(model)
serializers.save_npz(args.fused_model, fused_model)