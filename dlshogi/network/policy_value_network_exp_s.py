import torch
import torch.nn as nn
import torch.nn.functional as F


from dlshogi.common import *




class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))


    def forward(self, x):
        return x + self.bias




# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)




class ResNetBlock(nn.Module):
    def __init__(self, channels, activation):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = activation


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)


        out = self.conv2(out)
        out = self.bn2(out)


        return self.act(out + x)




class AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, activation):
        super(AttentionBlock, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.depth = d_model // nhead
        self.scale = self.depth ** -0.5

        self.qkv = nn.Conv2d(d_model, 3 * d_model, kernel_size=1)
        self.rel_bias = nn.Parameter(torch.zeros([1, 81, 81]))

        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)

        self.act = activation

        self.conv1 = nn.Conv2d(d_model, d_model * 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(d_model * 4)
        self.conv2 = nn.Conv2d(d_model * 4, d_model, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(d_model)


    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.split((self.d_model, self.d_model, self.d_model), dim=1)


        q = q.view(-1, self.nhead, self.depth, 81).transpose(2, 3)
        k = k.view(-1, self.nhead, self.depth, 81)
        v = v.view(-1, self.nhead, self.depth, 81)


        scores = torch.matmul(q, k) * self.scale + self.rel_bias

        attention = F.softmax(scores, dim=-1)

        out = torch.matmul(v, attention.transpose(2, 3))

        out = out.view(-1, self.d_model, 9, 9)

        out = self.proj(out)

        x = out + x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)


        out = self.conv2(out)
        out = self.bn2(out)


        return self.act(out + x)




class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks, channels, activation=nn.ReLU(), fcl=256):
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(
            in_channels=FEATURES1_NUM,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.l1_1_2 = nn.Conv2d(
            in_channels=FEATURES1_NUM,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.l1_2 = nn.Conv2d(
            in_channels=FEATURES2_NUM, out_channels=channels, kernel_size=1, bias=False
        )  # pieces_in_hand
        self.norm1_1_1 = nn.BatchNorm2d(channels)
        self.norm1_1_2 = nn.BatchNorm2d(channels)
        self.norm1_2 = nn.BatchNorm2d(channels)
        self.act = activation


        # Resnet blocks
        self.blocks = nn.Sequential(
            *[ResNetBlock(channels, activation) for _ in range(blocks - 2)]
        )

        self.attn_block = AttentionBlock(channels, 4, activation)


        # policy network
        self.policy = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = Bias(9 * 9 * MAX_MOVE_LABEL_NUM)


        # value network
        self.value_conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=MAX_MOVE_LABEL_NUM,
            kernel_size=1,
            bias=False,
        )
        self.value_norm1 = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9 * 9 * MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)


    def forward(self, x1, x2):
        u1_1_1 = self.norm1_1_1(self.l1_1_1(x1))
        u1_1_2 = self.norm1_1_2(self.l1_1_2(x1))
        u1_2 = self.norm1_2(self.l1_2(x2))
        u1 = self.act(u1_1_1 + u1_1_2 + u1_2)


        # resnet blocks
        h = self.blocks(u1)

        h = self.attn_block(h)


        # policy network
        h_policy = self.policy(h)
        h_policy = self.policy_bias(torch.flatten(h_policy, 1))


        # value network
        h_value = self.act(self.value_norm1(self.value_conv1(h)))
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)


        return h_policy, h_value


    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        activation = nn.SiLU() if memory_efficient else Swish()
        for n, m in self.named_modules():
            if isinstance(m, PolicyValueNetwork) or isinstance(m, ResNetBlock):
                m.act = activation
