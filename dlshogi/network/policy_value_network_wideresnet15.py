import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.bias

k = 192
fcl = 256 # fully connected layers
class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=k, kernel_size=1, bias=False) # pieces_in_hand
        self.l2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l3 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l4 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l5 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l6 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l7 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l8 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l9 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l10 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l11 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l12 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l13 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l14 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l15 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l16 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l17 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l18 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l19 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l20 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l21 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l22 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l23 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l24 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l25 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l26 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l27 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l28 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l29 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l30 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l31 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)

        # policy network
        self.l32 = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l32_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)

        # value network
        self.l32_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1)
        self.l33_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.l34_v = nn.Linear(fcl, 1)

        self.norm1 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm2 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm3 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm4 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm5 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm6 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm7 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm8 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm9 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm10 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm11 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm12 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm13 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm14 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm15 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm16 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm17 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm18 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm19 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm20 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm21 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm22 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm23 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm24 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm25 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm26 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm27 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm28 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm29 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm30 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm31 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm32_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM, eps=2e-05)

    def __call__(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = u1_1_1 + u1_1_2 + u1_2

        # Residual block
        h1 = F.relu(self.norm1(u1))
        h2 = F.relu(self.norm2(self.l2(h1)))
        u3 = self.l3(h2) + u1
        # Residual block
        h3 = F.relu(self.norm3(u3))
        h4 = F.relu(self.norm4(self.l4(h3)))
        u5 = self.l5(h4) + u3
        # Residual block
        h5 = F.relu(self.norm5(u5))
        h6 = F.relu(self.norm6(self.l6(h5)))
        u7 = self.l7(h6) + u5
        # Residual block
        h7 = F.relu(self.norm7(u7))
        h8 = F.relu(self.norm8(self.l8(h7)))
        u9 = self.l9(h8) + u7
        # Residual block
        h9 = F.relu(self.norm9(u9))
        h10 = F.relu(self.norm10(self.l10(h9)))
        u11 = self.l11(h10) + u9
        # Residual block
        h11 = F.relu(self.norm11(u11))
        h12 = F.relu(self.norm12(self.l12(h11)))
        u13 = self.l13(h12) + u11
        # Residual block
        h13 = F.relu(self.norm13(u13))
        h14 = F.relu(self.norm14(self.l14(h13)))
        u15 = self.l15(h14) + u13
        # Residual block
        h15 = F.relu(self.norm15(u15))
        h16 = F.relu(self.norm16(self.l16(h15)))
        u17 = self.l17(h16) + u15
        # Residual block
        h17 = F.relu(self.norm17(u17))
        h18 = F.relu(self.norm18(self.l18(h17)))
        u19 = self.l19(h18) + u17
        # Residual block
        h19 = F.relu(self.norm19(u19))
        h20 = F.relu(self.norm20(self.l20(h19)))
        u21 = self.l21(h20) + u19
        # Residual block
        h21 = F.relu(self.norm21(u21))
        h22 = F.relu(self.norm22(self.l22(h21)))
        u23 = self.l23(h22) + u21
        # Residual block
        h23 = F.relu(self.norm23(u23))
        h24 = F.relu(self.norm24(self.l24(h23)))
        u25 = self.l25(h24) + u23
        # Residual block
        h25 = F.relu(self.norm25(u25))
        h26 = F.relu(self.norm26(self.l26(h25)))
        u27 = self.l27(h26) + u25
        # Residual block
        h27 = F.relu(self.norm27(u27))
        h28 = F.relu(self.norm28(self.l28(h27)))
        u29 = self.l29(h28) + u27
        # Residual block
        h29 = F.relu(self.norm29(u29))
        h30 = F.relu(self.norm30(self.l30(h29)))
        u31 = self.l31(h30) + u29

        h31 = F.relu(self.norm31(u31))

        # policy network
        h32 = self.l32(h31)
        h32_1 = self.l32_2(h32.view(-1, 9*9*MAX_MOVE_LABEL_NUM))

        # value network
        h32_v = F.relu(self.norm32_v(self.l32_v(h31)))
        h33_v = F.relu(self.l33_v(h32_v.view(-1, 9*9*MAX_MOVE_LABEL_NUM)))
        return h32_1, self.l34_v(h33_v)