import torch
import torch.nn as nn

from dlshogi.common import *

# from https://github.com/deepmind/deepmind-research/tree/master/nfnets
_nonlin_gamma = dict(
    identity=1.0,
    celu=1.270926833152771,
    elu=1.2716004848480225,
    gelu=1.7015043497085571,
    leaky_relu=1.70590341091156,
    log_sigmoid=1.9193484783172607,
    log_softmax=1.0002083778381348,
    relu=1.7139588594436646,
    relu6=1.7131484746932983,
    selu=1.0008515119552612,
    sigmoid=4.803835391998291,
    silu=1.7881293296813965,
    softsign=2.338853120803833,
    softplus=1.9203323125839233,
    tanh=1.5939117670059204,
)

class WSConv2d(nn.Conv2d):
    """2D Convolution with Scaled Weight Standardization and affine gain+bias."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, gamma=1.0):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.gain = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        self.scale = gamma * self.weight[0].numel() ** -0.5 # gamma * 1 / sqrt(fan-in)

        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')
        if bias:
            nn.init.zeros_(self.bias)

    def standardize_weight(self, eps=1e-4):
        std, mean = torch.std_mean(self.weight, dim=(1, 2, 3), keepdims=True, unbiased=False)
        weight = self.scale * (self.weight - mean) / (std + eps)
        return self.gain * weight

    def forward(self, x):
        weight = self.standardize_weight()
        return torch.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class NFResBlock(nn.Module):
    """Normalizer-Free pre-activation ResNet Block."""

    def __init__(self, num_filters, activation, beta, alpha, gamma):
        super(NFResBlock, self).__init__()
        self.activation = activation
        self.beta, self.alpha = beta, alpha

        self.conv1 = WSConv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1, gamma=gamma)
        self.conv2 = WSConv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1, gamma=gamma)
        self.skip_gain = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.activation(x) * self.beta
        shortcut = x

        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(x)

        # SkipInit Gain
        out.mul_(self.skip_gain)

        out = out * self.alpha + shortcut
        return out

class PolicyValueNetwork(nn.Module):
    def __init__(self, num_blocks=10, num_filters=192, num_units=256, act='silu'):
        super(PolicyValueNetwork, self).__init__()
        
        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'silu':
            self.activation = nn.SiLU()

        self.conv1_1_1 = WSConv2d(in_channels=FEATURES1_NUM, out_channels=num_filters, kernel_size=3, padding=1, gamma=_nonlin_gamma[act])
        self.conv1_1_2 = WSConv2d(in_channels=FEATURES1_NUM, out_channels=num_filters, kernel_size=1, padding=0, gamma=_nonlin_gamma[act])
        self.conv1_2 = WSConv2d(in_channels=FEATURES2_NUM, out_channels=num_filters, kernel_size=1, gamma=_nonlin_gamma[act]) # pieces_in_hand

        # residual blocks
        blocks = []
        alpha = 0.2
        expected_std = 1.0
        for _ in range(num_blocks):
            # Scalar pre-multiplier so each block sees an N(0,1) input at init
            beta = 1./ expected_std
            blocks.append(NFResBlock(num_filters, self.activation, beta, alpha, gamma=_nonlin_gamma[act]))
            expected_std = (expected_std**2 + alpha**2)**0.5
        self.blocks = nn.Sequential(*blocks)

        # policy head
        self.policy_conv1 = nn.Conv2d(in_channels=num_filters, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias1 = nn.Parameter(torch.zeros(9*9*MAX_MOVE_LABEL_NUM))

        # value head
        self.value_conv1 = WSConv2d(in_channels=num_filters, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, gamma=_nonlin_gamma[act])
        self.value_fc1 = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, num_units)
        self.value_fc2 = nn.Linear(num_units, 1)

    def __call__(self, x1, x2):
        u1_1_1 = self.conv1_1_1(x1)
        u1_1_2 = self.conv1_1_2(x1)
        u1_2 = self.conv1_2(x2)
        x = u1_1_1 + u1_1_2 + u1_2

        # residual blocks
        x = self.blocks(x)
        x = self.activation(x)

        # policy network
        h_p = self.policy_conv1(x)
        policy = h_p.view(-1, 9*9*MAX_MOVE_LABEL_NUM) + self.policy_bias1

        # value network
        h_v = self.activation(self.value_conv1(x))
        h_v = self.activation(self.value_fc1(h_v.view(-1, 9*9*MAX_MOVE_LABEL_NUM)))
        value = self.value_fc2(h_v)

        return policy, value

    def agc_targets(self):
        targets = []
        for m in self.modules():
            if isinstance(m, WSConv2d):
                targets.extend(m.parameters())
        return targets
