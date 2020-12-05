import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *

class TreePolicy(nn.Module):
    def __init__(self):
        super(TreePolicy, self).__init__()
        self.fc1 = nn.Linear(9*9*FEATURES1_NUM + FEATURES2_NUM, 9*9*MAX_MOVE_LABEL_NUM, bias=False)

    def __call__(self, x):
        return self.fc1(x)
