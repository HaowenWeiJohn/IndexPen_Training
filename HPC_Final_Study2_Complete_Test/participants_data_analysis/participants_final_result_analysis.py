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


sys.argv.append('participant_2')
sys.argv.append('session_3')
# sys.argv.append('1')

argv_len = sys.argv
print('Number of arguments:', argv_len, 'arguments.')
print('Argument List:', str(sys.argv))

participant_name = sys.argv[1]
session_name = sys.argv[2]
# offline_evaluation = sys.argv[3]

random_state = 3

raw_acc_evaluation_dir = '../participants_session_raw_acc_evaluation'
####################################### raw accuracy for every session  #################
# dir: participants_session_raw_acc_evaluation
participant_raw_acc_evaluation_dir = os.path.join(raw_acc_evaluation_dir, participant_name)

original_model_acc = []
transfer_model_acc = []
transfer_fresh_model_acc = []
session_index = []

plt.rcParams['xtick.labelsize'] = 35
plt.rcParams['ytick.labelsize'] = 35
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['axes.titlesize'] = 40

for dirs in os.listdir(participant_raw_acc_evaluation_dir):
    acc_info_path = os.path.join(participant_raw_acc_evaluation_dir, dirs, 'transfer_learning_best_cm_hist_dict')
    with open(acc_info_path, 'rb') as f:
        original_model_cm, original_session_test_acc, \
        transfer_model_cm, transfer_session_test_acc, \
        transfer_fresh_model_cm, transfer_fresh_session_test_acc \
                    = pickle.load(f)
    original_model_acc.append(original_session_test_acc)
    transfer_model_acc.append(transfer_session_test_acc)
    transfer_fresh_model_acc.append(transfer_fresh_session_test_acc)
    session_index.append('S'+dirs.split('_')[-1])


fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot(111)

ax.plot(session_index, original_model_acc, linewidth=5, markersize=120, label='Base Model')
ax.plot(session_index, transfer_model_acc, linewidth=5, markersize=120, label='Transfer Model')
ax.plot(session_index, transfer_fresh_model_acc, linewidth=5, markersize=120, label='Transfer Fresh Model')
ax.grid()
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# Put a legend below current axis
ax.legend(fontsize=25, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)

ax.set_ylim([-0.05, 1])
ax.set_xlabel('Session')
ax.set_ylabel('Accuracy')
ax.set_title(participant_name)
# ax.grid()
# plt.ylim(0, 1.1)
plt.show()


####################################### lavenshtin distance between two sentences after removing Act, Enter, and Nois #################
# dir: participants_session_raw_prediction_evaluation
raw_prediction_evaluation_dir = '../participants_session_raw_prediction_evaluation'
participant_raw_prediction_evaluation_dir = os.path.join(raw_prediction_evaluation_dir, participant_name)

original_levenshtein_distances = []
transfer_levenshtein_distances = []
transfer_fresh_levenshtein_distances = []
for dirs in os.listdir(participant_raw_prediction_evaluation_dir):
    raw_prediction_evaluation_info_path = os.path.join(participant_raw_prediction_evaluation_dir, dirs, 'raw_prediction_evaluation')
    with open(raw_prediction_evaluation_info_path, 'rb') as f:
        detection_results, original_transfer_lite_model_prediction_result = pickle.load(f)

    session_original_lsd = []
    session_transfer_lsd = []
    session_transfer_fresh_lsd = []
    for trail in detection_results:
        trail_original_lsd = levenshtein_distance(detection_results[trail][1][0], detection_results[trail][0][0])
        session_original_lsd.append(trail_original_lsd)
        trail_transfer_lsd = levenshtein_distance(detection_results[trail][2][0], detection_results[trail][0][0])
        session_transfer_lsd.append(trail_transfer_lsd)
        trail_transfer_fresh_lsd = levenshtein_distance(detection_results[trail][3][0], detection_results[trail][0][0])
        session_transfer_fresh_lsd.append(trail_transfer_fresh_lsd)

    original_levenshtein_distances.append(np.mean(session_original_lsd))
    transfer_levenshtein_distances.append(np.mean(session_transfer_lsd))
    transfer_fresh_levenshtein_distances.append(np.mean(session_transfer_fresh_lsd))

fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot(111)

ax.plot(session_index, original_levenshtein_distances, linewidth=5, markersize=120, label='Base Model')
ax.plot(session_index, transfer_levenshtein_distances, linewidth=5, markersize=120, label='Transfer Model')
ax.plot(session_index, transfer_fresh_levenshtein_distances, linewidth=5, markersize=120, label='Transfer Fresh Model')
ax.grid()
# ax.legend(fontsize=60, loc='center left', bbox_to_anchor=(1, 0.5))

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
# Put a legend below current axis
ax.legend(fontsize=25, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)

ax.set_ylim([-0.05, 1])
ax.set_xlabel('Session')
ax.set_ylabel('String Similarity')
ax.set_title(participant_name)
# ax.grid()
# plt.ylim(0, 1.1)
plt.show()


    # detection result : ground truth, original model, lite model\


## generate some plots?

