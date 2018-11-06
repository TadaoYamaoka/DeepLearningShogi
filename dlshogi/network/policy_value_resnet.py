import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
# from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization, Add
from tensorflow.keras.layers import Input, Activation, Flatten
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

import shogi
# import shogi.CSA

# import argparse
# import random
# import copy

# import logging
# import os

from dlshogi.common import *

class Bias(Layer):

    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[1:]),
                                 initializer='uniform',
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
        # k = 256
        k = 192
        # model = Sequential()
        # # layer1
        # # model.add(Conv2D(k, (3, 3), padding='same', data_format='channels_first', input_shape=((len(shogi.PIECE_TYPES) + sum(shogi.MAX_PIECES_IN_HAND))*2+1, 9, 9)))
        # model.add(Conv2D(k, (3, 3), padding='same', data_format='channels_first', input_shape=(104, 9, 9)))
        # model.add(BatchNormalization(axis=1))
        # model.add(Activation('relu'))
        # # layer2 - 12
        # for i in range(11):
        #     model.add(Conv2D(k, (3, 3), padding='same', data_format='channels_first'))
        #     if i < 8:
        #         model.add(BatchNormalization(axis=1))
        #     model.add(Activation('relu'))
        # 
        # # policy network
        # # layer13
        # model.add(Conv2D(MOVE_DIRECTION_LABEL_NUM, (1, 1), data_format='channels_first', use_bias=False))
        # model.add(Flatten())
        # # model.add(Bias())
        # # model.add(Activation('softmax'))
        # # value network
        # model.add(Dense(units=256, activation='relu', input_dim=NUM_CLASSES))
        # model.add(Dense(units=1, activation="tanh", input_dim=256))
        
        main_input = Input(shape=(104, 9, 9))
        
        # # layer1
        # x = Conv2D(k, (3, 3), padding='same', data_format='channels_first')(main_input)
        # x = BatchNormalization(axis=1)(x)
        # x = Activation('relu')(x)
        # 
        # # layer2 - 12
        # for i in range(11):
        #     x = Conv2D(k, (3, 3), padding='same', data_format='channels_first')(x)
        #     if i < 8:
        #         x = BatchNormalization(axis=1)(x)
        #     x = Activation('relu')(x)
        
        x = Conv2D(k, (3, 3), padding='same', data_format='channels_first')(main_input)
        for i in range(11):
            x = self.block(k, x)
        
        # policy network
        # layer13
        ph = Conv2D(MOVE_DIRECTION_LABEL_NUM, (1, 1), data_format='channels_first', use_bias=False)(x)
        ph = Flatten()(ph)
        ph = Bias()(ph)
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
