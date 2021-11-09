import datetime
import glob
import os.path
import pickle
import shutil
import warnings
from copy import copy

import matplotlib
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
    1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 16, 18, 19, 20, 21
]

raw_acc_evaluation_dir = '../../HPC_Final_Study2_Complete_Test_Class_Balance_run_all/participants_session_raw_acc_evaluation'
raw_prediction_evaluation_dir = '../../HPC_Final_Study2_Complete_Test_Class_Balance_run_all/participants_session_raw_prediction_evaluation'

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

original_model_lsd_df = pd.DataFrame(columns=session_name)
transfer_model_lsd_df = pd.DataFrame(columns=session_name)
transfer_fresh_model_lsd_df = pd.DataFrame(columns=session_name)

# plot participant x

participants_lsd_dfs = {}
for participant_id in range(1, len(participant_ids)+1):
    for session in range(1,6):
        original_model_lsd_df['S'+str(session)] = participant_complete_result[participant_id][1][session]['original']
        transfer_model_lsd_df['S'+str(session)] = participant_complete_result[participant_id][1][session]['transfer']
        transfer_fresh_model_lsd_df['S'+str(session)] = participant_complete_result[participant_id][1][session]['fresh']

    participants_lsd_dfs[participant_id] = copy(transfer_model_lsd_df)



plot_groups = [(1,2,3),(4,5,6),(7,8,9),(10,11,12),(13,14,15,16)]

for plot_group in plot_groups:
    plt.rcParams['xtick.labelsize'] = 35
    plt.rcParams['ytick.labelsize'] = 35
    plt.rcParams['axes.labelsize'] = 45
    plt.rcParams['axes.titlesize'] = 45


    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111)

    legend_patch = []
    plot_lsd_select = plot_group
    for index, subject_index in enumerate(plot_lsd_select):
        lsd_df = participants_lsd_dfs[subject_index]
        lsd_df.boxplot(column=list(lsd_df.columns), sym='C' + str(subject_index),
                              color='C' + str(index+1))
        mean = lsd_df.median(axis=0)

        plt.plot(np.linspace(1.0, 5, 5), mean, alpha=.5, color='C'+str(index+1), marker="^", markersize=20)

        legend_patch.append(matplotlib.patches.Patch(color='C' + str(index+1), label='P 2-' + str(subject_index)))




    plt.legend(handles=legend_patch, loc='lower right', fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim((0, 1))


    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.xlabel('Session', fontsize=18)
    plt.ylabel('String Similarity', fontsize=18)
    # plt.title('Transfer Learning 20 Sample/Class 10 Fold', fontsize=20)

    plt.savefig('lsd_'+str(plot_group[0])+'-'+str(plot_group[-1]),dpi=300)
    plt.show()



# participants_last_session_lsd_avg = []
# for participant in participants_lsd_dfs:
#     last_session_average = np.mean(participants_lsd_dfs[participant]['S5'])
#     participants_last_session_lsd_avg.append(last_session_average)
#
# print(np.mean(participants_last_session_lsd_avg))

