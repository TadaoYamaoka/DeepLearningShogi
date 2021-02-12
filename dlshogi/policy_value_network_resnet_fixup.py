import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *

class FixupBlock(nn.Module):
    def __init__(self, filters):
        super(FixupBlock, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1, bias=False)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1, bias=False)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.conv1(x + self.bias1a)
        out = torch.relu_(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        out += x
        out = torch.relu_(out)

        return out

class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks=10, filters=192, units=256):
        super(PolicyValueNetwork, self).__init__()
        self.conv1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=filters, kernel_size=3, padding=1, bias=False)
        self.conv1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=filters, kernel_size=1, padding=0, bias=False)
        self.conv1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=filters, kernel_size=1, bias=False) # pieces_in_hand
        self.bias1 = nn.Parameter(torch.zeros(1))

        # residual blocks
        self.blocks = nn.Sequential(*[FixupBlock(filters) for _ in range(blocks)])

        # policy network
        self.policy_conv1 = nn.Conv2d(in_channels=filters, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias1 = nn.Parameter(torch.zeros(9*9*MAX_MOVE_LABEL_NUM))

        # value network
        self.value_conv1 = nn.Conv2d(in_channels=filters, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1)
        self.value_fc1 = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, units)
        self.value_fc2 = nn.Linear(units, 1)

        # fixup initialization 
        for m in self.modules():
            if isinstance(m, FixupBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * blocks ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
        nn.init.constant_(self.value_fc2.weight, 0)
        nn.init.constant_(self.value_fc2.bias, 0)

    def __call__(self, x1, x2):
        u1_1_1 = self.conv1_1_1(x1)
        u1_1_2 = self.conv1_1_2(x1)
        u1_2 = self.conv1_2(x2)
        x = torch.relu_(u1_1_1 + u1_1_2 + u1_2 + self.bias1)

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
