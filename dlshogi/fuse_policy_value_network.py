import numpy as np
import torch
import torch.nn as nn
from dlshogi import serializers
from dlshogi.policy_value_network import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('fused_model')
args = parser.parse_args()

def fuse_conv_and_bn(conv, bn):
    # init
    fusedconv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(b_conv + b_bn)

    return fusedconv

class FusedPolicyValueNetwork(nn.Module):
    def __init__(self, model):
        super(FusedPolicyValueNetwork, self).__init__()
        self.l1_1_1 = model.l1_1_1
        self.l1_1_2 = model.l1_1_2
        self.l1_2 = model.l1_2
        self.l2 = fuse_conv_and_bn(model.l2, model.norm2)
        self.l3 = model.l3
        self.l4 = fuse_conv_and_bn(model.l4, model.norm4)
        self.l5 = model.l5
        self.l6 = fuse_conv_and_bn(model.l6, model.norm6)
        self.l7 = model.l7
        self.l8 = fuse_conv_and_bn(model.l8, model.norm8)
        self.l9 = model.l9
        self.l10 = fuse_conv_and_bn(model.l10, model.norm10)
        self.l11 = model.l11
        self.l12 = fuse_conv_and_bn(model.l12, model.norm12)
        self.l13 = model.l13
        self.l14 = fuse_conv_and_bn(model.l14, model.norm14)
        self.l15 = model.l15
        self.l16 = fuse_conv_and_bn(model.l16, model.norm16)
        self.l17 = model.l17
        self.l18 = fuse_conv_and_bn(model.l18, model.norm18)
        self.l19 = model.l19
        self.l20 = fuse_conv_and_bn(model.l20, model.norm20)
        self.l21 = model.l21
        # policy network
        self.l22 = model.l22
        self.l22_2 = model.l22_2
        # value network
        self.l22_v = model.l22_v
        self.l23_v = model.l23_v
        self.l24_v = model.l24_v
        self.norm1 = model.norm1
        self.norm3 = model.norm3
        self.norm5 = model.norm5
        self.norm7 = model.norm7
        self.norm9 = model.norm9
        self.norm11 = model.norm11
        self.norm13 = model.norm13
        self.norm15 = model.norm15
        self.norm17 = model.norm17
        self.norm19 = model.norm19
        self.norm21 = model.norm21
        self.norm22_v = model.norm22_v

torch.set_grad_enabled(False)

model = PolicyValueNetwork()
serializers.load_npz(args.model, model)
model.eval()

fused_model= FusedPolicyValueNetwork(model)
serializers.save_npz(args.fused_model, fused_model)