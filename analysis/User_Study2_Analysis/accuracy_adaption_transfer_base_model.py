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
import pandas as pd
import sys
from analysis.User_Study2_Analysis.info_extraction_utils import extract_participant_info

# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training')

from data_utils.data_config import *
from data_utils.make_model import *
from data_utils.ploting import *
from data_utils.data_config import *
from data_utils.prediction_utils import *

######


participant_ids = [
    1, 2, 4, 5, 7, 8, 9, 11, 13, 16, 18, 19
]

raw_acc_evaluation_dir = '../../HPC_Final_Study2_Complete_Test_Class_Balance/participants_session_raw_acc_evaluation'
raw_prediction_evaluation_dir = '../../HPC_Final_Study2_Complete_Test_Class_Balance/participants_session_raw_prediction_evaluation'

participant_complete_result = {}

for index in range(0, len(participant_ids)):
    participant_id = participant_ids[index]
    participant_result = extract_participant_info(participant_id, raw_acc_evaluation_dir, raw_prediction_evaluation_dir)

    participant_complete_result[index + 1] = participant_result
    print(participant_id)

####################################################################################################
# make f1 plot for each participant
participant_each_plot = 5
session_name = ['S1', 'S2', 'S3', 'S4', 'S5']

original_model_f1_df = pd.DataFrame(columns=session_name)
transfer_model_f1_df = pd.DataFrame(columns=session_name)
transfer_fresh_model_f1_df = pd.DataFrame(columns=session_name)

for index in range(1, len(participant_ids)):
    participant_result = participant_complete_result[index]

    original_model_f1 = []
    transfer_model_f1 = []
    transfer_fresh_model_f1 = []

    for session in range(1, 6):
        original_session_test_f1 = f1_score(y_true=participant_result[0][session]['original'][0],
                                            y_pred=participant_result[0][session]['original'][1], average='macro')
        transfer_session_test_f1 = f1_score(y_true=participant_result[0][session]['transfer'][0],
                                            y_pred=participant_result[0][session]['transfer'][1], average='macro')
        transfer_fresh_session_test_f1 = f1_score(y_true=participant_result[0][session]['fresh'][0],
                                                  y_pred=participant_result[0][session]['fresh'][1], average='macro')

        original_model_f1.append(original_session_test_f1)
        transfer_model_f1.append(transfer_session_test_f1)
        transfer_fresh_model_f1.append(transfer_fresh_session_test_f1)

        # element wise f1


    original_model_f1_df.loc['P 2-' + str(index)] = original_model_f1
    transfer_model_f1_df.loc['P 2-' + str(index)] = transfer_model_f1
    transfer_fresh_model_f1_df.loc['P 2-' + str(index)] = transfer_fresh_model_f1


################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################

# calculate element wise f1 for all the chars

char_column_name = ['A', 'B', 'C', 'D', 'E',
                    'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O',
                    'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y',
                    'Z', 'Spc', 'Bspc', 'Ent', 'Act', 'Nois']

char_f1 = None

for index in range(1, len(participant_ids)):

    participant_original_model_char_f1_df = pd.DataFrame(columns=char_column_name)
    participant_transfer_model_char_f1_df = pd.DataFrame(columns=char_column_name)
    participant_transfer_fresh_char_model_f1_df = pd.DataFrame(columns=char_column_name)

    participant_result = participant_complete_result[index]

    for session in range(1, 6):
        original_session_test_f1 = f1_score(y_true=participant_result[0][session]['original'][0],
                                            y_pred=participant_result[0][session]['original'][1], average=None)
        transfer_session_test_f1 = f1_score(y_true=participant_result[0][session]['transfer'][0],
                                            y_pred=participant_result[0][session]['transfer'][1], average=None)
        transfer_fresh_session_test_f1 = f1_score(y_true=participant_result[0][session]['fresh'][0],
                                                  y_pred=participant_result[0][session]['fresh'][1], average=None)

        participant_original_model_char_f1_df.loc['session' + str(session)] = original_session_test_f1
        participant_transfer_model_char_f1_df.loc['session' + str(session)] = transfer_session_test_f1
        participant_transfer_fresh_char_model_f1_df.loc['session' + str(session)] = transfer_fresh_session_test_f1

    if char_f1 is None:
        char_f1 = np.expand_dims(np.array(participant_transfer_model_char_f1_df), axis=0)
    else:
        char_f1 = np.append(char_f1,np.expand_dims(np.array(participant_transfer_model_char_f1_df), axis=0), axis=0)

average_char_f1 = np.average(char_f1, axis=0)
average_char_f1 = np.transpose(average_char_f1)
average_char_f1_df = pd.DataFrame(data=average_char_f1, index=indexpen_classes, columns=session_name)


# element wise f1

# plot
plt.rcParams['xtick.labelsize'] = 35
plt.rcParams['ytick.labelsize'] = 35
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['axes.titlesize'] = 40



# def plot_dataframe_group_box(data_frame, plot_group=range(0,4),
#                           xlabel='Session', ylabel='f1 score'):
#
#     plt.rcParams['xtick.labelsize'] = 35
#     plt.rcParams['ytick.labelsize'] = 35
#     plt.rcParams['axes.labelsize'] = 40
#     plt.rcParams['axes.titlesize'] = 40
#
#
#     fig = plt.figure(figsize=(20, 15))
#     ax = fig.add_subplot(111)
#     for row in plot_group:
#         ax.plot(data_frame.iloc[row],
#                 linewidth=8,
#                 markersize=35,
#                 label=data_frame.index.values[row],
#                 marker='^')
#         ax.grid()
#         ax.legend(fontsize=40, loc='lower right')
#
#     ax.set_ylim([0, 1])
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     plt.grid(linestyle='-', linewidth=2.5)
#     # # ax.set_title('None')
#     # #     # ax.grid()
#     # # plt.ylim(0, 1.1)
#     plt.show()
#     return ax, fig


char_split_group = [range(0,5), range(5,10), range(10,15), range(15,20), range(20,25), range(25,31)]

for char_group in char_split_group:
    ax, fig = plot_dataframe_group_line(average_char_f1_df, plot_group=char_group, ylabel='cross user average f1 score')
    file_name = 'average_char_f1_'+\
                char_column_name[char_group[0]]+'_'+\
                char_column_name[char_group[-1]]
    fig.savefig(os.path.join('char_f1', file_name)
                , dpi=300)

char_split_group = [range(0,11)]

for char_group in char_split_group:
    ax, fig = plot_dataframe_group_line(transfer_model_f1_df, plot_group=char_group, ylabel='Average F-1 Score', ncol=3)
    file_name = 'transfer_model_f1_'+str(char_group[1]+1)+'_'+str(char_group[-1]+1)
    fig.savefig(os.path.join('user_f1/transfer_model', file_name)
                , dpi=300)
for char_group in char_split_group:
    ax, fig = plot_dataframe_group_line(original_model_f1_df, plot_group=char_group, ylabel='Average F-1 Score', ncol=3)
    file_name = 'original_model_f1_'+str(char_group[1]+1)+'_'+str(char_group[-1]+1)
    fig.savefig(os.path.join('user_f1/original_model', file_name)
                , dpi=300)




# plot_group = range(0, 4)
#
#
# fig = plt.figure(figsize=(20, 15))
# ax = fig.add_subplot(111)
# for row in plot_group:
#     ax.plot(average_char_f1_df.iloc[row],
#             linewidth=8,
#             markersize=35,
#             label=average_char_f1_df.index.values[row],
#             marker='^')
#     ax.grid()
#     ax.legend(fontsize=40, loc='lower right')
#
#
# ax.set_ylim([0, 1])
# ax.set_xlabel('Session')
# ax.set_ylabel('f1 score')
# plt.grid(linestyle='-', linewidth=2.5)
# # # ax.set_title('None')
# # #     # ax.grid()
# # # plt.ylim(0, 1.1)
# plt.show()








# table view for session 5

session_5_char_f1 = char_f1[:,-1,:]

session_5_char_f1 = np.transpose(session_5_char_f1)

participant_char_average = np.expand_dims(np.average(session_5_char_f1, axis=0), axis=0)

session_5_f1 = np.append(session_5_char_f1, participant_char_average, axis=0)

session_5_f1 = np.append(session_5_f1, np.expand_dims(np.average(session_5_f1, axis=-1), axis=-1), axis=-1)



