from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np

import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import Sequential, Model, Input
from tensorflow.python.keras.layers import TimeDistributed, Conv2D, BatchNormalization, MaxPooling2D, Flatten, \
    concatenate, LSTM, Dropout, Dense
import os
from tensorflow import keras

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import pickle
from tensorflow.keras.callbacks import CSVLogger


def noise_augmentation(x, y, mean=0, std=10, augmentation_factor=10, min_threshold=None, max_threshold=None,
                       time_series=True):
    # self duplicate
    x = np.repeat(x, repeats=len(x) * augmentation_factor, axis=0)
    y = np.repeat(x, repeats=len(y) * augmentation_factor, axis=0)

    # augumentation
    for sample_index in range(1, len(x)):
        sample_shape = x[sample_index].shape()
        x[sample_index] = x[sample_index] + np.random.normal(mean, std, x[sample_index].shape())
        # thresholding x sample
        if min_threshold:
            x[sample_index][x[sample_index] <= max_threshold] = min_threshold

        if max_threshold:
            x[sample_index][x[sample_index] >= max_threshold] = max_threshold


    return x, y
