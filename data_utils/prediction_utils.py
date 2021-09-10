import datetime
import glob
import os.path
import pickle
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedShuffleSplit
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from collections import deque
import sys

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training')
from data_utils.data_config import *
from data_utils.make_model import *
from data_utils.ploting import *
from data_utils.data_config import *
from data_utils.data_general_utils import *


def realtime_simulation_raw_prediction(ra_map_series, rd_map_series,
                                       buffer_size,
                                       interpreter,
                                       ):
    rd_hist_buffer = deque(maxlen=buffer_size)
    ra_hist_buffer = deque(maxlen=buffer_size)
    relaxCounter = 0

    pred_prob_hist_buffer = None

    input1_index = interpreter.get_input_details()[0]["index"]
    input2_index = interpreter.get_input_details()[1]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    for index in range(0, len(ra_map_series)):
        # push in sample
        rd_hist_buffer.append(rd_map_series[index])
        ra_hist_buffer.append(ra_map_series[index])
        print(index)
        if rd_hist_buffer.__len__() == rd_hist_buffer.maxlen:

            interpreter.set_tensor(input1_index,
                                   np.expand_dims(np.array(rd_hist_buffer), 0).astype(
                                       np.float32))
            interpreter.set_tensor(input2_index,
                                   np.expand_dims(np.array(ra_hist_buffer), 0).astype(
                                       np.float32))

            # invoke_start_time = time.time()
            interpreter.invoke()
            # print('Invoking duration: ', invoke_start_time - time.time())
            output = interpreter.tensor(output_index)
            yPred = np.array(output()[0])

            # add to history

            if index == 120 - 1:
                pred_prob_hist_buffer = yPred
            else:
                pred_prob_hist_buffer = np.vstack((pred_prob_hist_buffer, yPred))

    pred_prob_hist_buffer = pred_prob_hist_buffer.transpose()
    # plotted_char_prob = []

    return pred_prob_hist_buffer


def realtime_simulation_debouncer(pred_prob_hist_buffer,
                                  lstm_offset=120,
                                  number_of_classes=31,
                                  debouncerFrameThreshold=50,
                                  debouncerProbThreshold=0.8,
                                  relaxPeriod=15,
                                  ):
    debouncer = np.zeros(number_of_classes)
    relaxCounter = 0
    detect_chars_buffer = []
    detect_chars_index_buffer = []

    for index in range(0, pred_prob_hist_buffer.shape[1]):
        if relaxCounter == relaxPeriod:
            yPred = pred_prob_hist_buffer[:, index]
            # yPred = yPred[0]
            # add 1 to the char that hit break indices
            breakIndices = np.argwhere(yPred >= debouncerProbThreshold)
            # debouncer[breakIndices[:, 1]] += 1
            for i, debouncer_value in enumerate(debouncer):
                if i in breakIndices:
                    debouncer[i] += 1
                else:
                    if debouncer[i] > 0:
                        debouncer[i] -= 1

            detects = np.argwhere(np.array(debouncer) >= debouncerFrameThreshold)

            # zero out the debouncer that inactivated for debouncerProbThreshold frames

            if len(detects) > 0:
                # print(detects)
                detect_char = indexpen_classes[detects[0][0]]
                # print(detect_char)
                detect_chars_buffer.append(detect_char)
                detect_chars_index_buffer.append(index)
                debouncer = np.zeros(31)
                relaxCounter = 0

        else:
            relaxCounter += 1

    detect_chars_index_buffer = np.array(detect_chars_index_buffer) + lstm_offset-1

    return detect_chars_buffer, detect_chars_index_buffer


def prediction_archive(ra_map_series, rd_map_series,
                       buffer_size,
                       interpreter,
                       number_of_classes=31,
                       debouncerFrameThreshold=50,
                       debouncerProbThreshold=0.8,
                       relaxPeriod=15,
                       inactivateClearThreshold=10,
                       existing_char_only=True
                       ):
    rd_hist_buffer = deque(maxlen=buffer_size)
    ra_hist_buffer = deque(maxlen=buffer_size)
    debouncer = np.zeros(number_of_classes)
    relaxCounter = 0

    detect_chars_buffer = []
    detect_chars_index_buffer = []
    pred_prob_hist_buffer = None

    input1_index = interpreter.get_input_details()[0]["index"]
    input2_index = interpreter.get_input_details()[1]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    for index in range(0, len(ra_map_series)):
        # push in sample
        rd_hist_buffer.append(rd_map_series[index])
        ra_hist_buffer.append(ra_map_series[index])
        print(index)
        if rd_hist_buffer.__len__() == rd_hist_buffer.maxlen:
            # start prediction
            if relaxCounter == relaxPeriod:

                # yPred = model.predict(
                #     [np.expand_dims(np.array(rd_hist_buffer), 0),
                #      np.expand_dims(np.array(ra_hist_buffer), 0)])

                interpreter.set_tensor(input1_index,
                                       np.expand_dims(np.array(rd_hist_buffer), 0).astype(
                                           np.float32))
                interpreter.set_tensor(input2_index,
                                       np.expand_dims(np.array(ra_hist_buffer), 0).astype(
                                           np.float32))

                # invoke_start_time = time.time()
                interpreter.invoke()
                # print('Invoking duration: ', invoke_start_time - time.time())
                output = interpreter.tensor(output_index)
                yPred = np.array(output()[0])

                # add to history

                if index == 120 + relaxPeriod - 1:
                    pred_prob_hist_buffer = yPred
                else:
                    # pred_prob_hist_buffer = np.append(pred_prob_hist_buffer, yPred, axis=0)
                    pred_prob_hist_buffer = np.vstack((pred_prob_hist_buffer, yPred))

                # yPred = yPred[0]
                # add 1 to the char that hit break indices
                breakIndices = np.argwhere(yPred >= debouncerProbThreshold)
                # debouncer[breakIndices[:, 1]] += 1
                for i, debouncer_value in enumerate(debouncer):
                    if i in breakIndices:
                        debouncer[i] += 1
                    else:
                        if debouncer[i] > 0:
                            debouncer[i] -= 1

                detects = np.argwhere(np.array(debouncer) >= debouncerFrameThreshold)

                # zero out the debouncer that inactivated for debouncerProbThreshold frames

                if len(detects) > 0:
                    print(detects)
                    detect_char = indexpen_classes[detects[0][0]]
                    print(detect_char)
                    detect_chars_buffer.append(detect_char)
                    detect_chars_index_buffer.append(index)
                    debouncer = np.zeros(31)
                    relaxCounter = 0

            else:
                relaxCounter += 1

    pred_prob_hist_buffer = pred_prob_hist_buffer.transpose()
    # plotted_char_prob = []

    return pred_prob_hist_buffer, detect_chars_buffer, detect_chars_index_buffer


def plot_realtime_simulation(pred_prob_hist_buffer, detect_chars_buffer, grdt_chars, existing_char_only=True):
    plotted_char_prob = []

    fig = plt.figure(figsize=(60, 10))
    ax = fig.add_subplot(111)

    for index, char in enumerate(indexpen_classes):
        if existing_char_only:
            if char in grdt_chars:
                ax.plot(pred_prob_hist_buffer[index], label=char)
                plotted_char_prob.append(indexpen_classes[index])
            else:
                pass
        else:
            ax.plot(pred_prob_hist_buffer[index], label=char)
            plotted_char_prob.append(indexpen_classes[index])

    # ax.legend(bbox_to_anchor=(1.1, 1), loc=5, borderaxespad=0.)
    ax.set_aspect(aspect=80)
    plt.show()


def levenshtein_distance(detect_chars_buffer, grdt_chars):
    remove_list = ['Act', 'Nois']
    for index in range(0, detect_chars_buffer):
        if detect_chars_buffer[index] in remove_list:
            detect_chars_buffer.pop(index)

    for index in range(0, grdt_chars):
        if grdt_chars[index] in remove_list:
            grdt_chars.pop(index)

    special_replacement = {'Spc': '1', 'Bspc': '2'}

    pred_string = replace_special(''.join(detect_chars_buffer), special_replacement)
    grdt_string = replace_special(''.join(grdt_chars), special_replacement)

    str_dist = levenshtein_ratio_and_distance(pred_string, grdt_string, ratio_calc=True)

    return str_dist
