import numpy as np
import chainer
from chainer import cuda, Variable
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer import static_code
from chainer import static_graph

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
            l22=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l23=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l24=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l25=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l26=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l27=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l28=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l29=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l30=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),
            l31=L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1, nobias = True),

            # policy network
            l32=L.Convolution2D(in_channels = k, out_channels = MAX_MOVE_LABEL_NUM, ksize = 1, nobias = True),
            l32_2=L.Bias(shape=(9*9*MAX_MOVE_LABEL_NUM)),

            # value network
            l32_v=L.Convolution2D(in_channels = k, out_channels = MAX_MOVE_LABEL_NUM, ksize = 1),
            l33_v=L.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl),
            l34_v=L.Linear(fcl, 1),

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
            norm22=L.BatchNormalization(k),
            norm23=L.BatchNormalization(k),
            norm24=L.BatchNormalization(k),
            norm25=L.BatchNormalization(k),
            norm26=L.BatchNormalization(k),
            norm27=L.BatchNormalization(k),
            norm28=L.BatchNormalization(k),
            norm29=L.BatchNormalization(k),
            norm30=L.BatchNormalization(k),
            norm31=L.BatchNormalization(k),
            norm32_v=L.BatchNormalization(MAX_MOVE_LABEL_NUM)
        )

    @static_graph
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
        # Residual block
        h21 = F.relu(self.norm21(u21))
        h22 = F.dropout(F.relu(self.norm22(self.l22(h21))), ratio=dropout_ratio)
        u23 = self.l23(h22) + u21
        # Residual block
        h23 = F.relu(self.norm23(u23))
        h24 = F.dropout(F.relu(self.norm24(self.l24(h23))), ratio=dropout_ratio)
        u25 = self.l25(h24) + u23
        # Residual block
        h25 = F.relu(self.norm25(u25))
        h26 = F.dropout(F.relu(self.norm26(self.l26(h25))), ratio=dropout_ratio)
        u27 = self.l27(h26) + u25
        # Residual block
        h27 = F.relu(self.norm27(u27))
        h28 = F.dropout(F.relu(self.norm28(self.l28(h27))), ratio=dropout_ratio)
        u29 = self.l29(h28) + u27
        # Residual block
        h29 = F.relu(self.norm29(u29))
        h30 = F.dropout(F.relu(self.norm30(self.l30(h29))), ratio=dropout_ratio)
        u31 = self.l31(h30) + u29

        h31 = F.relu(self.norm31(u31))

        # policy network
        h32 = self.l32(h31)
        h32_1 = self.l32_2(F.reshape(h32, (len(h32.data), 9*9*MAX_MOVE_LABEL_NUM)))

        # value network
        h32_v = F.relu(self.norm32_v(self.l32_v(h31)))
        h33_v = F.relu(self.l33_v(h32_v))
        return h32_1, self.l34_v(h33_v)