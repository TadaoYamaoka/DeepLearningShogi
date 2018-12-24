import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization, Add
from tensorflow.keras.layers import Input, Activation, Flatten
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

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

class PolicyValueResnet():
    def __init__(self):
        self.model = self._build_model()
    
    def predict(self, x):
        return self.model.predict(x)
    
    def block(self, channles, input_tensor):
        # ショートカット元
        shortcut = input_tensor
        # メイン側
        x = BatchNormalization()(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(channles, (3, 3), padding='same', data_format='channels_first')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(channles, (3, 3), padding='same', data_format='channels_first')(x)
        # 結合
        return Add()([x, shortcut])
    
    def _build_model(self):
        k = 192
        main_input = Input(shape=(104, 9, 9))
        
        # layer1
        x = Conv2D(k, (3, 3), padding='same', data_format='channels_first')(main_input)
        
        # layer2 - 12
        for i in range(11):
            x = self.block(k, x)
        
        # policy network
        # layer13
        ph = Conv2D(MOVE_DIRECTION_LABEL_NUM, (1, 1), data_format='channels_first', use_bias=False)(x)
        ph = Flatten()(ph)
        ph = Bias(name = 'policy_head_notop')(ph)
        ph = Activation('softmax', name = 'policy_head')(ph)

        # value network
        # layer13
        vh = Conv2D(MOVE_DIRECTION_LABEL_NUM, (1, 1), data_format='channels_first', use_bias=False)(x)
        vh = Flatten()(vh)
        vh = Dense(units=256, activation='relu', input_dim=NUM_CLASSES)(vh)
        vh = Dense(units=1, activation="tanh", input_dim=256, name = 'value_head')(vh)

        model = Model(inputs=main_input, outputs=[ph, vh])
        
        return model

if __name__ == '__main__':
    network = PolicyValueResnet()
    model = network.model
