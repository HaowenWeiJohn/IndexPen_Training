import datetime
import glob
import os.path
import pickle
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedShuffleSplit
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import pandas as pd
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
'''
# user directory structure
    user1:
        session1_result:
            confusion matrix png
            loss png
            acc png
            history pickle

        session2_result:
            .............
            .............        
'''

''' 
    Data format:
        data/IndexPenData/IndexPenStudyData/UserStudy2Data
            User1:
                Session1 pickle
                    {
                    sample:
                        round1:
                            sentence0. {samples}
                            sentence1.
                            sentence2.
                            sentence3.
                            sentence4.
                        round2:
                            sentence5.
                            sentence6.
                            sentence7.
                            sentence8.
                            sentence9.

                    raw:
                            sentence0. {raw time series +-4 sec before and after the  session}
                            sentence1.
                            sentence2.
                            sentence3.
                            sentence4.
                        round2:
                            sentence5.
                            sentence6.
                            sentence7.
                            sentence8.
                            sentence9.
                    }
                Session2 pickle


    Coding task
        transfer_learning:
            train: first 5 sentences of that session and all the data before that session for the user
            test: last 5 sentences of that session                  
            evaluation:
                1. evaluate the normalized accuracy
                2. raw
'''

# load current session data

sys.argv.append('participant_1')
sys.argv.append('session_5')
# sys.argv.append('1')

argv_len = sys.argv
print('Number of arguments:', argv_len, 'arguments.')
print('Argument List:', str(sys.argv))

participant_name = sys.argv[1]
session_name = sys.argv[2]
print(participant_name, ' ', session_name, ' session_raw_acc_evaluation')
# offline_evaluation = sys.argv[3]


random_state = 3

original_model_path = indexpen_study2_original_model_path
original_lite_model_path = indexpen_study2_original_lite_model_path

load_data_dir = '../../data/IndexPenData/IndexPenStudyData/UserStudy2Data/'
participant_data_dir = os.path.join(load_data_dir, participant_name)

transfer_result_save_dir = '../participants_session_transfer_train'
session_index = int(session_name.split("_")[-1])
participant_session_transfer_train_dir = os.path.join(transfer_result_save_dir, participant_name,
                                                      'session_' + str(session_index - 1))

transfer_fresh_model_result_save_dir = '../participants_session_transfer_train_fresh_model'
session_index = int(session_name.split("_")[-1])
participant_session_transfer_train_fresh_model_dir = os.path.join(transfer_fresh_model_result_save_dir, participant_name,
                                                      'session_' + str(session_index - 1))

# the evaluation dir with model should be the previous session
evaluation_result_save_dir = '../participants_session_raw_acc_evaluation'
participant_raw_acc_evaluation_dir = os.path.join(evaluation_result_save_dir, participant_name)
participant_session_raw_acc_evaluation_dir = os.path.join(participant_raw_acc_evaluation_dir, session_name)


if session_index == 1:
    transfer_model_path = original_model_path
    transfer_lite_model_path = original_lite_model_path
    transfer_fresh_model_path = indexpen_study2_fresh_model_path
    transfer_fresh_lite_model_path = indexpen_study2_fresh_lite_model_path
else:
    if not os.path.isdir(participant_session_transfer_train_dir)\
    or not os.path.isdir(participant_session_transfer_train_fresh_model_dir):
        print('Please run the transfer learning model first')
        sys.exit(0)
    transfer_model_path = os.path.join(participant_session_transfer_train_dir, 'transfer_model.h5')
    transfer_lite_model_path = os.path.join(participant_session_transfer_train_dir,
                                            session_name+'_lite_models',
                                            'indexpen_model.tflite')

    transfer_fresh_model_path = os.path.join(participant_session_transfer_train_fresh_model_dir, 'transfer_model.h5')
    transfer_fresh_lite_model_path = os.path.join(participant_session_transfer_train_fresh_model_dir,
                                            session_name+'_lite_models',
                                            'indexpen_model.tflite')

# create participant evaluation file
if not os.path.isdir(participant_raw_acc_evaluation_dir):
    os.mkdir(participant_raw_acc_evaluation_dir)

if os.path.isdir(participant_session_raw_acc_evaluation_dir):
    shutil.rmtree(participant_session_raw_acc_evaluation_dir)
os.mkdir(participant_session_raw_acc_evaluation_dir)

transfer_model = tf.keras.models.load_model(transfer_model_path)
transfer_fresh_model = tf.keras.models.load_model(transfer_fresh_model_path)
original_model = tf.keras.models.load_model(original_model_path)


error_notes = pd.read_csv(os.path.join(indexpen_study2_error_recording_dir,
                                       participant_name+
                                       '_error_recording_form.csv'), index_col=0)
train_trails = {}
trail_index = 0

with open(os.path.join(participant_data_dir, session_name), 'rb') as f:
    participant_dir, session_dir, this_session_data = pickle.load(f)
if participant_dir != participant_name or session_dir != 'session_' + str(session_index):
    print('Data Session Error!')
    sys.exit(-1)
# remove error frame using the csv file
for error_target_trail in this_session_data:
    trail_error_list = error_notes.loc[str(error_target_trail) + '_error', :]
    trail_error_list_gt = error_notes.loc[str(error_target_trail), :]
    for index, char_index in enumerate(this_session_data[int(error_target_trail)][1][2]):
        char = indexpen_classes[int(char_index - 1)]
        if char != trail_error_list_gt[index]:
            print('Mismatching error recording and ground truth!')

    for error_sample_index in range(0, len(trail_error_list)):
        if pd.isnull(trail_error_list[error_sample_index]) is False:
            print('find error sample: trail ', str(error_target_trail), ' sample ',
                  str(trail_error_list_gt[error_sample_index]))

    error_index_list = np.where(np.array(trail_error_list) == 'X')[0]
    print(error_index_list)
    this_session_data[int(error_target_trail)][0][0][0] = \
        np.delete(this_session_data[int(error_target_trail)][0][0][0], error_index_list, axis=0)
    this_session_data[int(error_target_trail)][0][0][1] = \
        np.delete(this_session_data[int(error_target_trail)][0][0][1], error_index_list, axis=0)

    this_session_data[int(error_target_trail)][0] = list(this_session_data[int(error_target_trail)][0])
    this_session_data[int(error_target_trail)][0][1] = \
        np.delete(this_session_data[int(error_target_trail)][0][1], error_index_list, axis=0)
    this_session_data[int(error_target_trail)][0] = tuple(this_session_data[int(error_target_trail)][0])

# with open(os.path.join(participant_data_dir, session_name), 'rb') as f:
#     participant_dir, session_dir, evaluate_session_data = pickle.load(f)
# if participant_dir != participant_name or session_dir != session_name:
#     print('Data Session Error!')
#     sys.exit(-1)

X_mmw_rD_evaluate = []
X_mmw_rA_evaluate = []
Y_evaluate = []

# for trail in train_trails:
#     if len(X_mmw_rD) == 0:
#         X_mmw_rD = train_trails[trail][0][0][0]
#         X_mmw_rA = train_trails[trail][0][0][1]
#         Y = train_trails[trail][0][1]
#     else:
#         X_mmw_rD = np.concatenate([X_mmw_rD, train_trails[trail][0][0][0]])
#         X_mmw_rA = np.concatenate([X_mmw_rA, train_trails[trail][0][0][1]])
#         Y = np.concatenate([Y, train_trails[trail][0][1]])


for evaluate_trail in train_trails:
    if len(X_mmw_rD_evaluate) == 0:
        X_mmw_rD_evaluate = train_trails[evaluate_trail][0][0][0]
        X_mmw_rA_evaluate = train_trails[evaluate_trail][0][0][1]
        Y_evaluate = train_trails[evaluate_trail][0][1]
    else:
        X_mmw_rD_evaluate = np.concatenate([X_mmw_rD_evaluate, train_trails[evaluate_trail][0][0][0]])
        X_mmw_rA_evaluate = np.concatenate([X_mmw_rA_evaluate, train_trails[evaluate_trail][0][0][1]])
        Y_evaluate = np.concatenate([Y_evaluate, train_trails[evaluate_trail][0][1]])

################################

Y_original_pred1 = original_model.predict([X_mmw_rD_evaluate, X_mmw_rA_evaluate])
Y_original_pred_class = np.argmax(Y_original_pred1, axis=1)
Y_original_test_class = np.argmax(Y_evaluate, axis=1)

_, original_model_cm = plot_confusion_matrix(y_true=Y_original_test_class, y_pred=Y_original_pred_class,
                                             classes=indexpen_classes,
                                             normalize=False)

plt.savefig(os.path.join(participant_session_raw_acc_evaluation_dir, 'session_evaluation_confusion_matrix_original_model.png'))
plt.close()
plt.rcdefaults()

original_session_test_acc = accuracy_score(Y_original_test_class, Y_original_pred_class)
print("best_evaluate_session_accuracy_score_with_original_model:",
      original_session_test_acc)

###########################

Y_transfer_pred1 = transfer_model.predict([X_mmw_rD_evaluate, X_mmw_rA_evaluate])
Y_transfer_pred_class = np.argmax(Y_transfer_pred1, axis=1)
Y_transfer_test_class = np.argmax(Y_evaluate, axis=1)

_, transfer_model_cm = plot_confusion_matrix(y_true=Y_transfer_test_class, y_pred=Y_transfer_pred_class,
                                             classes=indexpen_classes,
                                             normalize=False)

plt.savefig(os.path.join(participant_session_raw_acc_evaluation_dir, 'session_evaluation_confusion_matrix_transfer_model.png'))
plt.close()
plt.rcdefaults()

transfer_session_test_acc = accuracy_score(Y_transfer_test_class, Y_transfer_pred_class)
print("best_evaluate_session_accuracy_score_with_transfer_model:",
      transfer_session_test_acc)

#######################

Y_transfer_fresh_pred1 = transfer_fresh_model.predict([X_mmw_rD_evaluate, X_mmw_rA_evaluate])
Y_transfer_fresh_pred_class = np.argmax(Y_transfer_fresh_pred1, axis=1)
Y_transfer_fresh_test_class = np.argmax(Y_evaluate, axis=1)

_, transfer_fresh_model_cm = plot_confusion_matrix(y_true=Y_transfer_fresh_test_class, y_pred=Y_transfer_fresh_pred_class,
                                             classes=indexpen_classes,
                                             normalize=False)

plt.savefig(os.path.join(participant_session_raw_acc_evaluation_dir, 'session_evaluation_confusion_matrix_transfer_fresh_model.png'))
plt.close()
plt.rcdefaults()

transfer_fresh_session_test_acc = accuracy_score(Y_transfer_fresh_test_class, Y_transfer_fresh_pred_class)
print("best_evaluate_session_accuracy_score_with_transfer_fresh_model:",
      transfer_fresh_session_test_acc)

#####################################
### realtime evaluation run all the sentences through original model and transfered model
###############################
# original_model_interpreter = tf.lite.Interpreter(original_lite_model_path)
# original_model_interpreter.allocate_tensors()
#
# transfer_model_interpreter = tf.lite.Interpreter(transfer_lite_model_path)
# transfer_model_interpreter.allocate_tensors()
#
#
# # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
# for evaluate_trail in evaluate_session_data:
#     rd_map_series = evaluate_session_data[evaluate_trail][1][0][0]
#     ra_map_series = evaluate_session_data[evaluate_trail][1][0][1]
#
#     labels_index = np.array(evaluate_session_data[evaluate_trail][1][2]).astype(int)-1
#     grdt_chars = np.array(indexpen_classes)[labels_index]
#     grdt_chars_index = evaluate_session_data[evaluate_trail][1][3]
#
#
#     pred_prob_hist_buffer, detect_chars_buffer, detect_chars_index_buffer= realtime_simulation(rd_map_series=rd_map_series,
#                         ra_map_series=ra_map_series,
#                         buffer_size=120,
#                         interpreter=original_model_interpreter)
#
#     plot_realtime_simulation(pred_prob_hist_buffer, detect_chars_buffer, detect_chars_index_buffer)
#



#####################################


with open(os.path.join(participant_session_raw_acc_evaluation_dir, 'transfer_learning_best_y_true_y_pred'), 'wb') as f:
    pickle.dump([
        [Y_original_test_class, Y_original_pred_class],
        [Y_transfer_test_class, Y_transfer_pred_class],
        [Y_transfer_fresh_test_class, Y_transfer_fresh_pred_class]
    ], f)

# run sentences through realtime debouncer algorithm