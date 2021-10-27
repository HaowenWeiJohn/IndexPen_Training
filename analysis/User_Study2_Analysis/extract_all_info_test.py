import datetime
import glob
import os.path
import pickle
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
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


sys.argv.append('participant_18')
participant_name = sys.argv[1]

random_state = 3

###
# data structure
# raw prediction:

###


raw_acc_evaluation_dir = '../../HPC_Final_Study2_Complete_Test_Class_Balance/participants_session_raw_acc_evaluation'
####################################### raw accuracy for every session  #################
# dir: participants_session_raw_acc_evaluation
participant_raw_acc_evaluation_dir = os.path.join(raw_acc_evaluation_dir, participant_name)

# original_model_acc = []
# transfer_model_acc = []
# transfer_fresh_model_acc = []
# session_index = []

raw_prediction_dict = {}

for dirs in os.listdir(participant_raw_acc_evaluation_dir):
    acc_info_path = os.path.join(participant_raw_acc_evaluation_dir, dirs, 'transfer_learning_best_y_true_y_pred')
    with open(acc_info_path, 'rb') as f:
        original_session_y_true_y_pred, \
        transfer_session_y_true_y_pred, \
        transfer_fresh_session_y_true_y_pred = \
            pickle.load(f)
    # cacluate f1 score
    # original_session_test_f1 = f1_score(y_true=original_session_y_true_y_pred[0],
    #                                     y_pred=original_session_y_true_y_pred[1], average='macro')
    # transfer_session_test_f1 = f1_score(y_true=transfer_session_y_true_y_pred[0],
    #                                     y_pred=transfer_session_y_true_y_pred[1], average='macro')
    # transfer_fresh_session_test_f1 = f1_score(y_true=transfer_fresh_session_y_true_y_pred[0],
    #                                           y_pred=transfer_fresh_session_y_true_y_pred[1], average='macro')

    # original_model_acc.append(original_session_test_f1)
    # transfer_model_acc.append(transfer_session_test_f1)
    # transfer_fresh_model_acc.append(transfer_fresh_session_test_f1)
    # session_index.append('S'+dirs.split('_')[-1])
    session_raw_prediction_dict = {
        'original': original_session_y_true_y_pred,
        'transfer': transfer_session_y_true_y_pred,
        'fresh': transfer_fresh_session_y_true_y_pred
    }

    raw_prediction_dict[int(dirs.split("_")[-1])] = session_raw_prediction_dict


####################################### lavenshtin distance between two sentences after removing Act, Enter, and Nois #################
# dir: participants_session_raw_prediction_evaluation
raw_prediction_evaluation_dir = '../../HPC_Final_Study2_Complete_Test_Class_Balance/participants_session_raw_prediction_evaluation'
participant_raw_prediction_evaluation_dir = os.path.join(raw_prediction_evaluation_dir, participant_name)

lsd_prediction = {}
for dirs in os.listdir(participant_raw_prediction_evaluation_dir):
    raw_prediction_evaluation_info_path = os.path.join(participant_raw_prediction_evaluation_dir, dirs,
                                                       'raw_prediction_evaluation')
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

    session_lsd_dict = {
        'original': session_original_lsd,
        'transfer': session_transfer_lsd,
        'fresh': session_transfer_fresh_lsd
    }

    lsd_prediction[int(dirs.split("_")[-1])] = session_lsd_dict




