import torch
import torch.nn as nn

from dlshogi.common import PIECETYPE_NUM, MAX_PIECES_IN_HAND_SUM

NUM_EMBEDDINGS1 = PIECETYPE_NUM * 2
NUM_EMBEDDINGS2 = MAX_PIECES_IN_HAND_SUM * 2 + 1

class LiteValueNetwork(nn.Module):
    def __init__(self, dims=(16, 4, 32), activation=nn.ReLU()):
        super(LiteValueNetwork, self).__init__()
        self.l1_1 = nn.Embedding(NUM_EMBEDDINGS1 + 1, dims[0], padding_idx=NUM_EMBEDDINGS1)
        self.l1_2 = nn.EmbeddingBag(NUM_EMBEDDINGS2 + 1, dims[0], mode='sum', padding_idx=NUM_EMBEDDINGS2)
        self.l2_1 = nn.Conv2d(in_channels=dims[0], out_channels=dims[1], kernel_size=(9, 1), bias=False)
        self.l2_2 = nn.Conv2d(in_channels=dims[0], out_channels=dims[1], kernel_size=(1, 9), bias=False)
        self.bn2_1 = nn.BatchNorm2d(dims[1])
        self.bn2_2 = nn.BatchNorm2d(dims[1])
        self.l3 = nn.Linear(dims[1] * 9 * 2, dims[2])
        self.l4 = nn.Linear(dims[2], 1)
        self.act = activation
        self.dims = dims

    def forward(self, x1, x2):
        h1_1 = self.l1_1(x1).view(-1, self.dims[0], 9, 9)
        h1_2 = self.l1_2(x2).view(-1, self.dims[0], 1, 1)
        h1 = h1_1 + h1_2
        h2_1 = self.bn2_1(self.l2_1(h1))
        h2_2 = self.bn2_2(self.l2_2(h1))
        h2 = self.act(torch.cat((h2_1.view(-1, self.dims[1] * 9), h2_2.view(-1, self.dims[1] * 9)), 1))
        h3 = self.act(self.l3(h2))
        h4 = self.l4(h3)

        return h4
