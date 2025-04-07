import torch
import torch.nn as nn

from dlshogi.common import FEATURES1_NUM, PIECETYPE_NUM, MAX_ATTACK_NUM, MAX_PIECES_IN_HAND_SUM

NUM_EMBEDDINGS1 = FEATURES1_NUM
NUM_EMBEDDINGS2 = MAX_PIECES_IN_HAND_SUM * 2 + 1
FEATURES1_LITE_NUM = 1 + 2 * (PIECETYPE_NUM + MAX_ATTACK_NUM)

class LiteValueNetwork(nn.Module):
    def __init__(self, dims=(16, 4, 32), activation=nn.ReLU()):
        super(LiteValueNetwork, self).__init__()
        self.l1_1 = nn.EmbeddingBag(NUM_EMBEDDINGS1 + 1, dims[0], mode='sum', padding_idx=NUM_EMBEDDINGS1)
        self.l1_2 = nn.EmbeddingBag(NUM_EMBEDDINGS2 + 1, dims[0], mode='sum', padding_idx=NUM_EMBEDDINGS2)
        self.bn1_1 = nn.BatchNorm2d(dims[0])
        self.bn1_2 = nn.BatchNorm2d(dims[0])
        self.l2 = nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=2, groups=dims[0], bias=False)
        self.bn2 = nn.BatchNorm2d(dims[0])
        self.l3_1 = nn.Conv2d(in_channels=dims[0], out_channels=dims[1], kernel_size=(8, 1), bias=False)
        self.l3_2 = nn.Conv2d(in_channels=dims[0], out_channels=dims[1], kernel_size=(1, 8), bias=False)
        self.bn3_1 = nn.BatchNorm2d(dims[1])
        self.bn3_2 = nn.BatchNorm2d(dims[1])
        self.l4 = nn.Linear(dims[1] * 8 * 2, dims[2])
        self.l5 = nn.Linear(dims[2], 1)
        self.act = activation
        self.dims = dims

    def forward(self, x1, x2):
        h1_1 = self.bn1_1(self.l1_1(x1.view(-1, FEATURES1_LITE_NUM)).view(-1, 9, 9, self.dims[0]).permute(0, 3, 1, 2))
        h1_2 = self.bn1_2(self.l1_2(x2).view(-1, self.dims[0], 1, 1))
        h1 = h1_1 + h1_2
        h2 = self.act(self.bn2(self.l2(h1)))
        h3_1 = self.bn3_1(self.l3_1(h2))
        h3_2 = self.bn3_2(self.l3_2(h2))
        h3 = self.act(torch.cat((h3_1.reshape(-1, self.dims[1] * 8), h3_2.reshape(-1, self.dims[1] * 8)), 1))
        h4 = self.act(self.l4(h3))
        h5 = self.l5(h4)

        return h5
