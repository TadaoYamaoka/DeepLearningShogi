import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Activation, Flatten
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

from dlshogi.common import *
from dlshogi.network.policy_value_resnet_with_logits import *
from dlshogi.features import *
from dlshogi.read_kifu import *

import shogi
import shogi.CSA

import argparse
import random
import copy

import logging
import os

parser = argparse.ArgumentParser(description='Deep Learning Shogi')
parser.add_argument('train_kifu_list', type=str, help='train kifu list')
parser.add_argument('test_kifu_list', type=str, help='test kifu list')
parser.add_argument('--batchsize', '-b', type=int, default=8, help='Number of positions in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given folder')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--model', default='model', help='The directory where the model and training/evaluation summaries are stored.')
parser.add_argument('--use_tpu', '-t', action='store_true', help='Use TPU model instead of CPU')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

logging.debug('read kifu start')
positions_train = read_kifu(args.train_kifu_list)
positions_test = read_kifu(args.test_kifu_list)
logging.debug('read kifu end')

logging.info('train position num = {}'.format(len(positions_train)))
logging.info('test position num = {}'.format(len(positions_test)))

# mini batch
def mini_batch(positions, i, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    mini_batch_win = []
    for b in range(batchsize):
        features, move, win = make_features(positions[i + b])
        mini_batch_data.append(features)
        mini_batch_move.append(move)
        mini_batch_win.append(win)

    return (np.array(mini_batch_data, dtype=np.float32),
            to_categorical(mini_batch_move, NUM_CLASSES),
            np.array(mini_batch_win, dtype=np.int32).reshape((-1, 1)))

# data generator
def datagen(positions):
    while True:
        positions_shuffled = random.sample(positions, len(positions))
        for i in range(0, len(positions_shuffled) - args.batchsize, args.batchsize):
            x, t1, t2 = mini_batch(positions_shuffled, i, args.batchsize)
            yield (x, {'policy_head': t1, 'value_head': t2})

def categorical_crossentropy(y_true, y_pred):
    return tf.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

def categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, tf.nn.softmax(y_pred))

if not os.path.isdir(args.model):
    os.mkdir(args.model)

if os.path.isfile(args.initmodel):
    model = load_model(args.initmodel, custom_objects={'Bias': Bias})
    
else:
    network = PolicyValueResnet()
    model = network.model

if args.use_tpu:
    optimizer = tf.train.MomentumOptimizer(0.001, momentum = 0.9)
else:
    optimizer = SGD(lr=0.001, momentum=0.9)

model.compile(loss={'policy_head': categorical_crossentropy, 'value_head': 'mean_squared_error'},
              optimizer=optimizer,
              loss_weights={'policy_head': 0.5, 'value_head': 0.5},
              metrics=['accuracy', categorical_accuracy])

checkpoint_path = args.model + "/model_policy_value_resnet_with_logits-best.hdf5"
checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True)

if args.use_tpu:
    # TPU
    import tensorflow as tf
    from tensorflow.contrib.tpu.python.tpu import keras_support
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

logging.info('Training start')
model.fit_generator(datagen(positions_train), int(len(positions_train) / args.batchsize),
          epochs=args.epoch,
          validation_data=datagen(positions_test), validation_steps=int(len(positions_test) / args.batchsize),
          callbacks=[checkpoint])
logging.info('Training end')

model_path = args.model + "/model_policy_value_resnet_with_logits-final.hdf5"
model.save_weights(model_path, save_format="h5")

if not args.use_tpu:
    model.save(args.model + "/model_policy_value_resnet_with_logits.h5")
    with open(args.model + "/model_policy_value_resnet_with_logits.json", "w") as fjson:
        fjson.write(model.to_json())
    tf.contrib.saved_model.save_keras_model(model, args.model)

import gc; gc.collect()
logging.info('Done')
