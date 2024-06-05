import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dlshogi.common import *

class PositionalEncoding(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = nn.Parameter(torch.zeros(1, channels, 9, 9))

    def forward(self, x):
        return x + self.pos_encoding
    
class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.bias

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)

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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, channels, d_model, nhead, dim_feedforward=256, dropout=0.1, activation=nn.GELU()):
        super(TransformerEncoderLayer, self).__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.depth = d_model // nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        self.qkv_linear = nn.Conv2d(channels, 3 * d_model, kernel_size=1, groups=nhead, bias=False)
        self.relative_linear1 = nn.Linear(channels * 81, 32, bias=False)
        self.relative_linear2 = nn.Linear(32, self.depth * 81, bias=False)
        self.o_linear = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)

        self.attention_dropout = nn.Dropout(dropout)
        self.linear1 = nn.Conv2d(d_model, dim_feedforward, kernel_size=1, bias=False)
        self.linear2 = nn.Conv2d(dim_feedforward, channels, kernel_size=1, bias=False)
        self.final_dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm2d(d_model)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        qkv = self.qkv_linear(x)
        q, k, v = qkv.split((self.d_model, self.d_model, self.d_model), dim=1)

        q = q.view(-1, self.nhead, self.depth, 81).transpose(2, 3)
        k = k.view(-1, self.nhead, self.depth, 81)
        v = v.view(-1, self.nhead, self.depth, 81).transpose(2, 3)

        r = self.relative_linear1(x.flatten(1))
        r = self.relative_linear2(r)
        r = r.view(-1, 1, self.depth, 81)
        r = torch.matmul(q, r)

        scores = (torch.matmul(q, k) + r) / math.sqrt(self.depth)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attended = torch.matmul(attention_weights, v)
        attended = attended.transpose(2, 3).contiguous().view(-1, self.d_model, 9, 9)
        attended = self.o_linear(attended)

        x = self.norm1(attended + x)
        feedforward = self.activation(self.linear1(x))
        feedforward = self.linear2(feedforward)
        feedforward = self.final_dropout(feedforward)
        x = self.norm2(feedforward + x)

        return x

class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks, channels, activation=nn.ReLU(), fcl=256, num_layers=8):
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=channels, kernel_size=1, bias=False) # pieces_in_hand
        self.norm1 = nn.BatchNorm2d(channels)
        self.act = activation

        # Resnet blocks
        self.blocks = nn.Sequential(*[ResNetBlock(channels, activation) for _ in range(blocks)])
        
        # Transformer
        self.pos_encoder = PositionalEncoding(channels)
        self.transformer = nn.Sequential(*[TransformerEncoderLayer(channels, d_model=256, nhead=8, dim_feedforward=256, dropout=0.1, activation=nn.GELU()) for _ in range(num_layers)])

        # policy network
        self.policy = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = Bias(9*9*MAX_MOVE_LABEL_NUM)

        # value network
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.value_norm1 = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = self.act(self.norm1(u1_1_1 + u1_1_2 + u1_2))

        # resnet blocks
        h = self.blocks(u1)
        
        # Transformer
        h = self.pos_encoder(h)
        h = self.transformer(h)
        
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
