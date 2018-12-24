from chainer import Chain
import chainer.functions as F
import chainer.links as L

from pydlshogi.common import *

ch = 192
fcl = 256
class PolicyValueNetwork(Chain):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        with self.init_scope():
            self.l1=L.Convolution2D(in_channels = 104, out_channels = ch, ksize = 3, pad = 1)
            self.l2=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.l3=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.l4=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.l5=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.l6=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.l7=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.l8=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.l9=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.l10=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.l11=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.l12=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            # policy network
            self.l13=L.Convolution2D(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1, nobias = True)
            self.l13_bias=L.Bias(shape=(9*9*MOVE_DIRECTION_LABEL_NUM))
            # value network
            self.l13_v=L.Convolution2D(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1)
            self.l14_v=L.Linear(9*9*MOVE_DIRECTION_LABEL_NUM, fcl)
            self.l15_v=L.Linear(fcl, 1)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        h5 = F.relu(self.l5(h4))
        h6 = F.relu(self.l6(h5))
        h7 = F.relu(self.l7(h6))
        h8 = F.relu(self.l8(h7))
        h9 = F.relu(self.l9(h8))
        h10 = F.relu(self.l10(h9))
        h11 = F.relu(self.l11(h10))
        h12 = F.relu(self.l12(h11))
        # policy network
        h13 = self.l13(h12)
        policy = self.l13_bias(F.reshape(h13, (-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))
        # value network
        h13_v = F.relu(self.l13_v(h12))
        h14_v = F.relu(self.l14_v(h13_v))
        value = self.l15_v(h14_v)
        return policy, value
