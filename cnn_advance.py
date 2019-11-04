# -*- coding: utf-8 -*-
# @Time     :11/4/19 11:56 AM
# @Auther   :Jason Lin
# @File     :cnn_advance.py
# @Software :PyCharm

import pandas as pd
import numpy as np
import pickle as pkl
import os
from sklearn.model_selection import StratifiedKFold
from keras.models import Model, model_from_json
from keras import layers
import keras
from sklearn import metrics
from scipy import interp
import matplotlib.pylab as plt

from keras.layers import Input, Dense, Reshape, Conv2D, Flatten, BatchNormalization, Activation
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

def cnn_std_advance():
    # X_train, y_train = load_data()
    inputs = Input(shape=(1, 23, 8), name='main_input')

    ###### reorganize the channel  ######
    conv_channel = Conv2D(16, (1, 1), padding='same')(inputs)
    conv_channel = BatchNormalization()(conv_channel)
    conv_channel = Activation('relu')(conv_channel)
    #############################

    conv_1 = Conv2D(10, (1, 1), padding='same')(conv_channel)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)

    conv_2 = Conv2D(10, (1, 2), padding='same')(conv_channel)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    conv_3 = Conv2D(10, (1, 3), padding='same')(conv_channel)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)

    conv_4 = Conv2D(10, (1, 5), padding='same')(conv_channel)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)

    conv_output = layers.concatenate([conv_1, conv_2, conv_3, conv_4])

    ###### reorganize the channel  ######
    conv_channel_out = Conv2D(16, (1, 1), padding='same')(conv_output)
    conv_channel_out = BatchNormalization()(conv_channel_out)
    conv_channel_out = Activation('relu')(conv_channel_out)
    #############################

    # bn_output = BatchNormalization()(conv_output)

    # pooling_output = layers.MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

    flatten_output = Flatten()(conv_channel_out)
    x = Dense(20, activation='relu')(flatten_output)
    x = layers.Dropout(rate=0.3)(x)

    prediction = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs, prediction)

    return model
