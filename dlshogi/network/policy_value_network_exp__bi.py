import torch
import torch.nn as nn
import torch.nn.functional as F


from dlshogi.common import *


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.bias


class ResNetBlock(nn.Module):
    def __init__(self, channels, activation, use_se=False):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = activation
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        return self.act(out + x)


class InceptionBlock(nn.Module):
    def __init__(self, channels, activation):
        super(InceptionBlock, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, channels, kernel_size=(9, 1), bias=False)
        self.conv1_2 = nn.Conv2d(channels, channels, kernel_size=(1, 9), bias=False)
        self.conv1_3 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels)
        self.bn1_2 = nn.BatchNorm2d(channels)
        self.bn1_3 = nn.BatchNorm2d(channels)
        self.conv2_1 = nn.Conv2d(channels, channels, kernel_size=(9, 1), bias=False)
        self.conv2_2 = nn.Conv2d(channels, channels, kernel_size=(1, 9), bias=False)
        self.conv2_3 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels)
        self.bn2_2 = nn.BatchNorm2d(channels)
        self.bn2_3 = nn.BatchNorm2d(channels)
        self.act = activation

    def forward(self, x):
        out = (
            self.bn1_1(self.conv1_1(x))
            + self.bn1_2(self.conv1_2(x))
            + self.bn1_3(self.conv1_3(x))
        )
        out = self.act(out)

        out = (
            self.bn2_1(self.conv2_1(out))
            + self.bn2_2(self.conv2_2(out))
            + self.bn2_3(self.conv2_3(out))
        )

        return self.act(out + x)


class AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, activation):
        super(AttentionBlock, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.depth = d_model // nhead
        self.scale = self.depth**-0.5

        # -----------------------------------------------------------
        # RMS-based QK normalization
        # q/k are produced as [B, d_model, 9, 9]. For attention we
        # reinterpret them as [B, nhead, depth, 81], so RMS normalization
        # must be applied over the per-head depth axis, not over batch,
        # token, or spatial axes. This keeps each square/head vector scale
        # stable before QK^T is computed.
        # -----------------------------------------------------------
        self.qk_norm_eps = 1e-6

        # -----------------------------------------------------------
        # Gating Mechanism Implementation
        # Paper: "Gated Attention for Large Language Models"
        # Method: SDPA Elementwise Gating (G1)
        # Position: After SDPA, before Output Projection
        # -----------------------------------------------------------
        self.qkv_gate = nn.Conv2d(d_model, 4 * d_model, kernel_size=1, bias=False)

        # -----------------------------------------------------------
        # Relative Positional Encoding Setup
        # -----------------------------------------------------------
        window_size = 9
        self.num_relative_distance = (2 * window_size - 1) * (2 * window_size - 1)
        # 学習可能なバイアステーブル
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(nhead, self.num_relative_distance)
        )
        # 相対位置インデックスの生成
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        # 2点間の相対座標を計算
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index_flat", relative_position_index.view(-1))
        # パラメータ初期化（小さな分散で初期化するのが一般的）
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(d_model)

        self.act = activation

        # -----------------------------------------------------------
        # Feed-Forward Network (FFN): SwiGLU
        #   hidden = 4 * d_model (same width as the previous FFN)
        #   SwiGLU(x) = (W_v x) * SiLU(W_g x)
        # -----------------------------------------------------------
        self.ffn_hidden = d_model * 4
        self.ffn_in = nn.Conv2d(d_model, 2 * self.ffn_hidden, kernel_size=1, bias=False)
        self.ffn_bn = nn.BatchNorm2d(2 * self.ffn_hidden)
        self.ffn_out = nn.Conv2d(self.ffn_hidden, d_model, kernel_size=1, bias=False)
        self.ffn_out_bn = nn.BatchNorm2d(d_model)
        self.swiglu_act = nn.SiLU()

    def _rms_qk_norm(self, x):
        # x: [B, nhead, depth, 81]
        # Normalize each head/square vector over depth. This is deliberately
        # not BatchNorm: no batch statistics and no mixing across board squares.
        rms = torch.rsqrt(x.pow(2).mean(dim=2, keepdim=True) + self.qk_norm_eps)
        return x * rms

    def forward(self, x):
        qkvg = self.qkv_gate(x)
        q, k, v, g = qkvg.split((self.d_model, self.d_model, self.d_model, self.d_model), dim=1)

        # Split heads first: [B, d_model, 9, 9] -> [B, nhead, depth, 81].
        # RMS-based QK norm is applied after q/k split and over depth only.
        q = q.view(-1, self.nhead, self.depth, 81)
        k = k.view(-1, self.nhead, self.depth, 81)
        q = self._rms_qk_norm(q)
        k = self._rms_qk_norm(k)

        q = q.transpose(2, 3)
        v = v.view(-1, self.nhead, self.depth, 81)

        # -----------------------------------------------------------
        # Apply Relative Positional Encoding
        # -----------------------------------------------------------
        relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index_flat]
        relative_position_bias = relative_position_bias.view(1, self.nhead, 81, 81)

        scores = torch.matmul(q, k) * self.scale + relative_position_bias

        attention = F.softmax(scores, dim=-1)

        out = torch.matmul(v, attention.transpose(2, 3))

        out = out.view(-1, self.d_model, 9, 9)

        # -----------------------------------------------------------
        # Apply Gating
        # Formula: Y' = Y * sigmoid(X * W_theta)
        # Y is 'out' (SDPA output), X is 'x' (Block input)
        # This introduces input-dependent sparsity
        # -----------------------------------------------------------
        gating_scores = torch.sigmoid(g)
        out = out * gating_scores

        out = self.proj(out)
        out = self.bn(out)

        x = out + x

        # -----------------------------------------------------------
        # FFN (SwiGLU)
        # -----------------------------------------------------------
        out = self.ffn_in(x)
        out = self.ffn_bn(out)
        gate, value = out.split((self.ffn_hidden, self.ffn_hidden), dim=1)
        out = value * self.swiglu_act(gate)

        out = self.ffn_out(out)
        out = self.ffn_out_bn(out)

        return self.act(out + x)


class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks, channels, activation=nn.ReLU(), fcl=256):
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(
            in_channels=FEATURES1_NUM,
            out_channels=channels,
            kernel_size=7,
            padding=3,
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

        # -----------------------------------------------------------
        # Absolute Positional Encoding Setup
        # -----------------------------------------------------------
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, channels, 9, 9))
        nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        # Resnet blocks
        _blocks = []
        for i in range(blocks):
            if i % 10 == 8:
                _blocks.append(AttentionBlock(channels, 4, activation))
            elif i % 10 == 9:
                pass
            elif i % 5 == 4:
                _blocks.append(InceptionBlock(channels, activation))
            elif i % 5 == 3:
                _blocks.append(ResNetBlock(channels, activation, use_se=True))
            else:
                _blocks.append(ResNetBlock(channels, activation))
        self.blocks = nn.Sequential(*_blocks)

        # policy network
        self.policy = nn.Conv2d(
            in_channels=channels,
            out_channels=MAX_MOVE_LABEL_NUM,
            kernel_size=1,
            bias=False,
        )
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
        u1 = self.act(u1_1_1 + u1_1_2 + u1_2 + self.absolute_pos_embed)

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
