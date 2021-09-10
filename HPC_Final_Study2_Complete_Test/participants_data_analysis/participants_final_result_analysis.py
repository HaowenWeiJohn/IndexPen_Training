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


sys.argv.append('participant_1')
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
session_index = []

plt.rcParams['xtick.labelsize'] = 35
plt.rcParams['ytick.labelsize'] = 35
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['axes.titlesize'] = 40

for dirs in os.listdir(participant_raw_acc_evaluation_dir):
    acc_info_path = os.path.join(participant_raw_acc_evaluation_dir, dirs, 'transfer_learning_best_cm_hist_dict')
    with open(acc_info_path, 'rb') as f:
        original_model_cm, original_session_test_acc, transfer_model_cm, transfer_session_test_acc = pickle.load(f)
    original_model_acc.append(original_session_test_acc)
    transfer_model_acc.append(transfer_session_test_acc)
    session_index.append('S'+dirs.split('_')[-1])


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)

ax.plot(session_index, original_model_acc, linewidth=5, markersize=120, label='Base Model')
ax.plot(session_index, transfer_model_acc, linewidth=5, markersize=120, label='Transfer Model')
ax.grid()
ax.legend(fontsize=30, loc='lower right')
ax.set_xlabel('Session')
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1])

# ax.grid()
# plt.ylim(0, 1.1)
plt.show()


####################################### lavenshtin distance between two sentences after removing Act, Enter, and Nois #################
# dir: participants_session_raw_acc_evaluation


## generate some plots?

