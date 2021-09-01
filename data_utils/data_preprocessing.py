from cv2 import cv2
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
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


def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

def noise_augmentation(x, y, mean=0, std=10, augmentation_factor=10, min_threshold=None, max_threshold=None,
                       time_series=True):
    # self duplicate
    print(len(x))
    x_repeat = np.repeat(x, repeats=augmentation_factor, axis=0)
    y_repeat = np.repeat(y, repeats=augmentation_factor, axis=0)

    # augumentation
    for sample_index in range(0, len(x_repeat)):
        sample_shape = x_repeat[sample_index].shape
        x_repeat[sample_index] = x_repeat[sample_index] + np.random.normal(mean, std, x_repeat[sample_index].shape)
        a = x_repeat[sample_index]
        # thresholding x sample
        if min_threshold:
            x_repeat[sample_index][x_repeat[sample_index] <= min_threshold] = min_threshold

        if max_threshold:
            x_repeat[sample_index][x_repeat[sample_index] >= max_threshold] = max_threshold

    x = np.concatenate((x, x_repeat))
    y = np.concatenate((y, y_repeat))

    return x, y


def dtw_image_archive(target_ts_img, known_ts_img, duration=120):
    # 120 * 8 * 16   120 * 8 * 16
    # for target_ts_data
    # (height, width, 3) # 3 is 3 channels for Red, Green, Blue
    total_distance = 0
    target_ts_img = target_ts_img.reshape((duration, -1))
    known_ts_img = known_ts_img.reshape((duration, -1))


    for channel in range(0, target_ts_img.shape[-1]):
        distance, paths = dtw.warping_paths(target_ts_img[:, channel], known_ts_img[:, channel])
        total_distance +=distance

    return total_distance


def dtw_rd(target_ts_img, known_ts_img, duration=120, filename="warp.png"):
    # 120 * 8 * 16   120 * 8 * 16
    # for target_ts_data
    # (height, width, 3) # 3 is 3 channels for Red, Green, Blue

    # target_ts_img = np.abs(target_ts_img)
    # known_ts_img = np.abs(known_ts_img)

    target_ts_img[target_ts_img < 0] = 0
    known_ts_img[known_ts_img < 0] = 0

    tar_neg_speed_avg = target_ts_img[:, :, 0:8].mean(axis=(1,2))
    tar_pos_speed_avg = target_ts_img[:, :, 8:16].mean(axis=(1,2))
    known_neg_speed_avg = known_ts_img[:, :, 0:8].mean(axis=(1,2))
    known_pos_speed_avg = known_ts_img[:, :, 8:16].mean(axis=(1,2))

    distance_neg, paths_neg = dtw.warping_paths(s1=tar_neg_speed_avg, s2=known_neg_speed_avg)
    distance_pos, paths_pos = dtw.warping_paths(s1=tar_pos_speed_avg, s2=known_pos_speed_avg)

    distance = distance_neg+distance_pos
    # for channel in range(0, target_ts_img.shape[-1]):
    #     distance, paths = dtw.warping_paths(target_ts_img[:, channel], known_ts_img[:, channel])
    #     total_distance += distance

    path = dtw.warping_path(tar_neg_speed_avg, known_neg_speed_avg)
    dtwvis.plot_warping(tar_neg_speed_avg.flatten(), known_neg_speed_avg.flatten(), path, filename=filename+'_neg')

    path = dtw.warping_path(tar_pos_speed_avg, known_pos_speed_avg)
    dtwvis.plot_warping(tar_pos_speed_avg.flatten(), known_pos_speed_avg.flatten(), path, filename=filename+'_pos')

    return distance


