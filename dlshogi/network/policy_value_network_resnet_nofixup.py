import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FixupBlock(nn.Module):
    def __init__(self, channels, activation):
        super(FixupBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.act = activation

    def forward(self, x):
        out = self.conv1(x)
        out = torch.relu_(out)

        out = self.conv2(out)

        out += x
        out = torch.relu_(out)

        return out

class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks, channels, activation=nn.ReLU(), fcl=256):
        super(PolicyValueNetwork, self).__init__()
        self.conv1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.conv1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=1, padding=0, bias=False)
        self.conv1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=channels, kernel_size=1, bias=False) # pieces_in_hand
        self.act = activation

        # residual blocks
        self.blocks = nn.Sequential(*[FixupBlock(channels, activation) for _ in range(blocks)])

        # policy network
        self.policy_conv1 = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias1 = nn.Parameter(torch.zeros(9*9*MAX_MOVE_LABEL_NUM))

        # value network
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1)
        self.value_fc1 = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x1, x2):
        u1_1_1 = self.conv1_1_1(x1)
        u1_1_2 = self.conv1_1_2(x1)
        u1_2 = self.conv1_2(x2)
        x = torch.relu_(u1_1_1 + u1_1_2 + u1_2)

        # residual blocks
        x = self.blocks(x)

        # policy network
        h_p = self.policy_conv1(x)
        policy = h_p.view(-1, 9*9*MAX_MOVE_LABEL_NUM) + self.policy_bias1

        # value network
        h_v = torch.relu_(self.value_conv1(x))
        h_v = torch.relu_(self.value_fc1(h_v.view(-1, 9*9*MAX_MOVE_LABEL_NUM)))
        value = self.value_fc2(h_v)

        return policy, value

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        activation = nn.SiLU() if memory_efficient else Swish()
        for n, m in self.named_modules():
            if isinstance(m, PolicyValueNetwork) or isinstance(m, FixupBlock):
                m.act = activation
