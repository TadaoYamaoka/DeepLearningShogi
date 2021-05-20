import torch
import torch.nn as nn

from dlshogi.common import *

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.bias

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, activation, kernel_size=3, stride=1, groups=1):
        super(ConvBN, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, channels, expand_ratio, activation):
        super(InvertedResidual, self).__init__()

        hidden_dim = int(round(channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBN(channels, hidden_dim, activation, kernel_size=1))
        layers.extend([
            # dw
            ConvBN(hidden_dim, hidden_dim, activation, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x)

class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks, channels, expand_ratio, activation=nn.ReLU(), fcl=256):
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=channels, kernel_size=1, bias=False) # pieces_in_hand
        self.norm1 = nn.BatchNorm2d(channels)
        self.act = activation

        # inverted residual blocks
        self.blocks = nn.Sequential(*[InvertedResidual(channels, expand_ratio, activation) for _ in range(blocks)])

        # policy network
        self.policy = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = Bias(9*9*MAX_MOVE_LABEL_NUM)

        # value network
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1)
        self.value_norm1 = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def __call__(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = self.act(self.norm1(u1_1_1 + u1_1_2 + u1_2))

        # inverted residual blocks
        h = self.blocks(u1)

        # policy network
        h_policy = self.policy(h)
        h_policy = self.policy_bias(h_policy.view(-1, 9*9*MAX_MOVE_LABEL_NUM))

        # value network
        h_value = self.act(self.value_norm1(self.value_conv1(h)))
        h_value = self.act(self.value_fc1(h_value.view(-1, 9*9*MAX_MOVE_LABEL_NUM)))
        h_value = self.value_fc2(h_value)

        return h_policy, h_value

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        for n, m in self.named_modules():
            if isinstance(m, PolicyValueNetwork) or isinstance(m, ConvBN):
                m.act = nn.SiLU() if memory_efficient else Swish()
