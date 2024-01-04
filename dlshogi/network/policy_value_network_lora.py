import math
import torch
import torch.nn as nn

from dlshogi.common import *
from dlshogi.network.policy_value_network_resnet import Bias, ResNetBlock


# LoRA wrapper for Conv2d
class LoRAConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, bias=False, rank=32):
        super(LoRAConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.U = nn.Conv2d(rank, out_channels, kernel_size, padding=padding, bias=False)
        self.V = nn.Conv2d(in_channels, rank, 1, bias=False)
        torch.nn.init.kaiming_uniform_(self.V.weight, a=math.sqrt(5))
        torch.nn.init.constant_(self.U.weight, 0)

    def forward(self, x):
        mid = self.V(x)
        return super().forward(x) + self.U(mid)



class LoRAResNetBlock(nn.Module):
    def __init__(self, channels, activation, lora_rank=32):
        super(LoRAResNetBlock, self).__init__()
        self.conv1 = LoRAConv2d(channels, channels, kernel_size=3, padding=1, bias=False, rank=lora_rank)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = LoRAConv2d(channels, channels, kernel_size=3, padding=1, bias=False, rank=lora_rank)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = activation

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return self.act(out + x)


class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks, channels, activation=nn.ReLU(), fcl=256, lora_rank=32):
        super(PolicyValueNetwork, self).__init__()

        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.act = activation

        # LoRA-enhanced ResNet blocks
        self.blocks = nn.Sequential(*[LoRAResNetBlock(channels, activation, lora_rank) for _ in range(blocks)])

        # policy network with LoRA
        self.policy = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = Bias(9*9*MAX_MOVE_LABEL_NUM)

        # value network with LoRA
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

        # policy network
        h_policy = self.policy(h)
        h_policy = self.policy_bias(torch.flatten(h_policy, 1))

        # value network
        h_value = self.act(self.value_norm1(self.value_conv1(h)))
        h_value = self.act(self.value_fc1(torch.flatten(h_value, 1)))
        h_value = self.value_fc2(h_value)

        return h_policy, h_value


def load_pretrained_model(pretrained_model_path, lora_model, map_location):
    # 学習済みのPolicyValueNetworkモデルのパラメータをロードする
    pretrained_state_dict = torch.load(pretrained_model_path, map_location=map_location)

    # LoRAモデルの現在のstate_dictを取得する
    lora_state_dict = lora_model.state_dict()

    # LoRAモデルのstate_dictを更新する
    for name, param in pretrained_state_dict["model"].items():
        if name in lora_state_dict:
            lora_state_dict[name].copy_(param)

    # LoRA層以外の全てのパラメータを固定する
    for name, param in lora_model.named_parameters():
        if "U" not in name and "V" not in name:  # LoRA層のパラメータは"U"と"V"を含む
            param.requires_grad = False
