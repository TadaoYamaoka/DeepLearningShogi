import torch
import torch.nn as nn

from dlshogi.common import *

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.bias

class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, activation, kernel):
        super(ConvNeXtBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel, padding=(kernel-1)//2, groups=channels, bias=True)
        self.bn1 = nn.GroupNorm(1, channels, eps=1e-6)
        self.conv2 = nn.Conv2d(channels, channels * 4, kernel_size=1, padding=0, bias=True)
        self.act = activation
        self.conv3 = nn.Conv2d(channels * 4, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.act(out)

        out = self.conv3(out)

        return out + x

class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks, channels, activation=nn.GELU(), fcl=256, kernel=3):
        super(PolicyValueNetwork, self).__init__()
        channels2 = channels // 8 * 3
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels2, kernel_size=3, padding=1, bias=True)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels2, kernel_size=1, padding=0, bias=True)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=channels2, kernel_size=1, bias=True) # pieces_in_hand
        self.norm1 = nn.GroupNorm(1, channels2, eps=1e-6)
        self.act = activation

        # Resnet blocks
        self.blocks = nn.Sequential(*[ConvNeXtBlock(channels2, activation, kernel) for _ in range(blocks)])

        # policy network
        self.policy = nn.Conv2d(in_channels=channels2, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = Bias(9*9*MAX_MOVE_LABEL_NUM)

        # value network
        self.value_conv1 = nn.Conv2d(in_channels=channels2, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=True)
        self.value_norm1 = nn.GroupNorm(1, MAX_MOVE_LABEL_NUM, eps=1e-6)
        self.value_fc1 = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = self.norm1(u1_1_1 + u1_1_2 + u1_2)

        # resnet blocks
        h = self.blocks(u1)

        # policy network
        h_policy = self.policy(h)
        h_policy = self.policy_bias(torch.flatten(h_policy, 1))

        # value network
        h_value = self.act(self.value_norm1(self.value_conv1(h)))
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)

        return h_policy, h_value
