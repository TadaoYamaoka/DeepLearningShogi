import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from dlshogi.common import *

k = 192
dropout_ratio = 0.1
fcl = 256 # fully connected layers
class PolicyValueNetwork(Chain):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__(
            l1_1_1=L.Convolution2D(in_channels = FEATURES1_NUM, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l1_1_2=L.Convolution2D(in_channels = FEATURES1_NUM, out_channels = k, ksize = 1, pad = 0, nobias = True),
            l1_2=L.Convolution2D(in_channels = FEATURES2_NUM, out_channels = k, ksize = 1, nobias = True), # pieces_in_hand
            l2=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l3=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l4=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l5=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l6=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l7=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l8=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l9=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l10=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l11=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l12=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l13=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l14=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l15=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l16=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l17=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l18=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l19=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l20=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l21=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            # policy network
            l22=L.Convolution2D(in_channels = k, out_channels = MAX_MOVE_LABEL_NUM, ksize = 1, nobias = True),
            l22_2=L.Bias(shape=(9*9*MAX_MOVE_LABEL_NUM)),
            # value network
            l22_v=L.Convolution2D(in_channels = k, out_channels = MAX_MOVE_LABEL_NUM, ksize = 1),
            l23_v=L.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl),
            l24_v=L.Linear(fcl, 1),
            norm1=L.BatchNormalization(k),
            norm2=L.BatchNormalization(k),
            norm3=L.BatchNormalization(k),
            norm4=L.BatchNormalization(k),
            norm5=L.BatchNormalization(k),
            norm6=L.BatchNormalization(k),
            norm7=L.BatchNormalization(k),
            norm8=L.BatchNormalization(k),
            norm9=L.BatchNormalization(k),
            norm10=L.BatchNormalization(k),
            norm11=L.BatchNormalization(k),
            norm12=L.BatchNormalization(k),
            norm13=L.BatchNormalization(k),
            norm14=L.BatchNormalization(k),
            norm15=L.BatchNormalization(k),
            norm16=L.BatchNormalization(k),
            norm17=L.BatchNormalization(k),
            norm18=L.BatchNormalization(k),
            norm19=L.BatchNormalization(k),
            norm20=L.BatchNormalization(k),
            norm21=L.BatchNormalization(k),
            norm22_v=L.BatchNormalization(MAX_MOVE_LABEL_NUM)
        )

    def __call__(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u1 = u1_1_1 + u1_1_2 + u1_2
        # Residual block
        h1 = F.relu(self.norm1(u1))
        h2 = F.dropout(F.relu(self.norm2(self.l2(h1))), ratio=dropout_ratio)
        u3 = self.l3(h2) + u1
        # Residual block
        h3 = F.relu(self.norm3(u3))
        h4 = F.dropout(F.relu(self.norm4(self.l4(h3))), ratio=dropout_ratio)
        u5 = self.l5(h4) + u3
        # Residual block
        h5 = F.relu(self.norm5(u5))
        h6 = F.dropout(F.relu(self.norm6(self.l6(h5))), ratio=dropout_ratio)
        u7 = self.l7(h6) + u5
        # Residual block
        h7 = F.relu(self.norm7(u7))
        h8 = F.dropout(F.relu(self.norm8(self.l8(h7))), ratio=dropout_ratio)
        u9 = self.l9(h8) + u7
        # Residual block
        h9 = F.relu(self.norm9(u9))
        h10 = F.dropout(F.relu(self.norm10(self.l10(h9))), ratio=dropout_ratio)
        u11 = self.l11(h10) + u9
        # Residual block
        h11 = F.relu(self.norm11(u11))
        h12 = F.dropout(F.relu(self.norm12(self.l12(h11))), ratio=dropout_ratio)
        u13 = self.l13(h12) + u11
        # Residual block
        h13 = F.relu(self.norm13(u13))
        h14 = F.dropout(F.relu(self.norm14(self.l14(h13))), ratio=dropout_ratio)
        u15 = self.l15(h14) + u13
        # Residual block
        h15 = F.relu(self.norm15(u15))
        h16 = F.dropout(F.relu(self.norm16(self.l16(h15))), ratio=dropout_ratio)
        u17 = self.l17(h16) + u15
        # Residual block
        h17 = F.relu(self.norm17(u17))
        h18 = F.dropout(F.relu(self.norm18(self.l18(h17))), ratio=dropout_ratio)
        u19 = self.l19(h18) + u17
        # Residual block
        h19 = F.relu(self.norm19(u19))
        h20 = F.dropout(F.relu(self.norm20(self.l20(h19))), ratio=dropout_ratio)
        u21 = self.l21(h20) + u19
        h21 = F.relu(self.norm21(u21))
        # policy network
        h22 = self.l22(h21)
        h22_1 = self.l22_2(F.reshape(h22, (len(h22.data), 9*9*MAX_MOVE_LABEL_NUM)))
        # value network
        h22_v = F.relu(self.norm22_v(self.l22_v(h21)))
        h23_v = F.relu(self.l23_v(h22_v))
        return h22_1, self.l24_v(h23_v)
