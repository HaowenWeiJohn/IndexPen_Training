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

# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training')

from data_utils.data_config import *
from data_utils.make_model import *
from data_utils.ploting import *
from data_utils.data_config import *
from data_utils.prediction_utils import *
######


with open('plot_result_save', 'rb') as f:
    pred_prob_hist_buffer, detect_chars_buffer, detect_chars_index_buffer, grdt_chars, grdt_chars_index = pickle.load(f)

plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['axes.titlesize'] = 40

fig = plt.figure(figsize=(50, 10))
ax = fig.add_subplot(111)

plotted_char_prob = []
existing_char_only = True
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
# ax.set_aspect(aspect=800)

'''
remove Nois?
'''
# for i in grdt_chars:
#     grdt_chars


ax.set_xticks(grdt_chars_index)
ax.set_xticklabels(grdt_chars)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

grdt_axis = ax.secondary_xaxis('top')
grdt_axis.set_xticks(detect_chars_index_buffer)
grdt_axis.set_xticklabels(detect_chars_buffer, color='r')
plt.setp(grdt_axis.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")



ax.yaxis.set_tick_params(labelsize=30)
ax.text(detect_chars_index_buffer, [1]*len(detect_chars_index_buffer), '')

plt.show()