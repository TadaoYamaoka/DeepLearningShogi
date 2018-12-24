import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Activation, Flatten
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense

import shogi

from dlshogi.common import *

class Bias(Layer):

    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[1:]),
                                 initializer='zeros',
                                 trainable=True)
        super(Bias, self).build(input_shape)

    def call(self, x):
        return x + self.W

class PolicyNetwork():
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        k = 192
        main_input = Input(shape=(104, 9, 9))
        
        # layer1
        x = Conv2D(k, (3, 3), padding='same', data_format='channels_first')(main_input)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)
        
        # layer2 - 12
        for i in range(11):
            x = Conv2D(k, (3, 3), padding='same', data_format='channels_first')(x)
            if i < 8:
                x = BatchNormalization(axis=1)(x)
            x = Activation('relu')(x)
        
        # policy network
        # layer13
        x = Conv2D(MOVE_DIRECTION_LABEL_NUM, (1, 1), data_format='channels_first', use_bias=False)(x)
        x = Flatten()(x)
        x = Bias(name = 'policy_head_notop')(x)
        x = Activation('softmax', name = 'policy_head')(x)
        
        model = Model(inputs=main_input, outputs=x)
        
        return model

if __name__ == '__main__':
    network = PolicyNetwork()
    model = network.model
