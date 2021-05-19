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

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


k = 224
fcl = 256 # fully connected layers
class PolicyValueNetwork(nn.Module):
    def __init__(self, use_aux=False):
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
        self.l32_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l33_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.l34_v = nn.Linear(fcl, 1)
        # sennichite, nyugyoku
        if use_aux:
            self.l34_aux = nn.Linear(fcl, 2)

        self.norm1 = nn.BatchNorm2d(k)
        self.norm2 = nn.BatchNorm2d(k)
        self.norm3 = nn.BatchNorm2d(k)
        self.norm4 = nn.BatchNorm2d(k)
        self.norm5 = nn.BatchNorm2d(k)
        self.norm6 = nn.BatchNorm2d(k)
        self.norm7 = nn.BatchNorm2d(k)
        self.norm8 = nn.BatchNorm2d(k)
        self.norm9 = nn.BatchNorm2d(k)
        self.norm10 = nn.BatchNorm2d(k)
        self.norm11 = nn.BatchNorm2d(k)
        self.norm12 = nn.BatchNorm2d(k)
        self.norm13 = nn.BatchNorm2d(k)
        self.norm14 = nn.BatchNorm2d(k)
        self.norm15 = nn.BatchNorm2d(k)
        self.norm16 = nn.BatchNorm2d(k)
        self.norm17 = nn.BatchNorm2d(k)
        self.norm18 = nn.BatchNorm2d(k)
        self.norm19 = nn.BatchNorm2d(k)
        self.norm20 = nn.BatchNorm2d(k)
        self.norm21 = nn.BatchNorm2d(k)
        self.norm22 = nn.BatchNorm2d(k)
        self.norm23 = nn.BatchNorm2d(k)
        self.norm24 = nn.BatchNorm2d(k)
        self.norm25 = nn.BatchNorm2d(k)
        self.norm26 = nn.BatchNorm2d(k)
        self.norm27 = nn.BatchNorm2d(k)
        self.norm28 = nn.BatchNorm2d(k)
        self.norm29 = nn.BatchNorm2d(k)
        self.norm30 = nn.BatchNorm2d(k)
        self.norm31 = nn.BatchNorm2d(k)
        self.norm32_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.swish = nn.SiLU()
        self.use_aux = use_aux

    def __call__(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = self.swish(self.norm1(u1_1_1 + u1_1_2 + u1_2))
        # Residual block
        h2 = self.swish(self.norm2(self.l2(u1)))
        h3 = self.norm3(self.l3(h2))
        u3 = self.swish(h3 + u1)
        # Residual block
        h4 = self.swish(self.norm4(self.l4(u3)))
        h5 = self.norm5(self.l5(h4))
        u5 = self.swish(h5 + u3)
        # Residual block
        h6 = self.swish(self.norm6(self.l6(u5)))
        h7 = self.norm7(self.l7(h6))
        u7 = self.swish(h7 + u5)
        # Residual block
        h8 = self.swish(self.norm8(self.l8(u7)))
        h9 = self.norm9(self.l9(h8))
        u9 = self.swish(h9 + u7)
        # Residual block
        h10 = self.swish(self.norm10(self.l10(u9)))
        h11 = self.norm11(self.l11(h10))
        u11 = self.swish(h11 + u9)
        # Residual block
        h12 = self.swish(self.norm12(self.l12(u11)))
        h13 = self.norm13(self.l13(h12))
        u13 = self.swish(h13 + u11)
        # Residual block
        h14 = self.swish(self.norm14(self.l14(u13)))
        h15 = self.norm15(self.l15(h14))
        u15 = self.swish(h15 + u13)
        # Residual block
        h16 = self.swish(self.norm16(self.l16(u15)))
        h17 = self.norm17(self.l17(h16))
        u17 = self.swish(h17 + u15)
        # Residual block
        h18 = self.swish(self.norm18(self.l18(u17)))
        h19 = self.norm19(self.l19(h18))
        u19 = self.swish(h19 + u17)
        # Residual block
        h20 = self.swish(self.norm20(self.l20(u19)))
        h21 = self.norm21(self.l21(h20))
        u21 = self.swish(h21 + u19)
        # Residual block
        h22 = self.swish(self.norm22(self.l22(u21)))
        h23 = self.norm23(self.l23(h22))
        u23 = self.swish(h23 + u21)
        # Residual block
        h24 = self.swish(self.norm24(self.l24(u23)))
        h25 = self.norm25(self.l25(h24))
        u25 = self.swish(h25 + u23)
        # Residual block
        h26 = self.swish(self.norm26(self.l26(u25)))
        h27 = self.norm27(self.l27(h26))
        u27 = self.swish(h27 + u25)
        # Residual block
        h28 = self.swish(self.norm28(self.l28(u27)))
        h29 = self.norm29(self.l29(h28))
        u29 = self.swish(h29 + u27)
        # Residual block
        h30 = self.swish(self.norm30(self.l30(u29)))
        h31 = self.norm31(self.l31(h30))
        u31 = self.swish(h31 + u29)
        # policy network
        h32 = self.l32(u31)
        h32_1 = self.l32_2(h32.view(-1, 9*9*MAX_MOVE_LABEL_NUM))
        # value network
        h32_v = self.swish(self.norm32_v(self.l32_v(u31)))
        h33_v = self.swish(self.l33_v(h32_v.view(-1, 9*9*MAX_MOVE_LABEL_NUM)))
        if self.use_aux:
            return h32_1, self.l34_v(h33_v), self.l34_aux(h33_v)
        else:
            return h32_1, self.l34_v(h33_v)


    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self.swish = nn.SiLU() if memory_efficient else Swish()
