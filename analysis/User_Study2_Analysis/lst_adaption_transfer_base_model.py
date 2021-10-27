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

original_model_lsd_df = pd.DataFrame(columns=session_name)
transfer_model_lsd_df = pd.DataFrame(columns=session_name)
transfer_fresh_model_lsd_df = pd.DataFrame(columns=session_name)

# plot participant x
participant_id = 2

for session in range(1,6):
    original_model_lsd_df['S'+str(session)] = participant_complete_result[participant_id][1][session]['original']
    transfer_model_lsd_df['S'+str(session)] = participant_complete_result[participant_id][1][session]['transfer']
    transfer_fresh_model_lsd_df['S'+str(session)] = participant_complete_result[participant_id][1][session]['fresh']

transfer_model_lsd_df.boxplot(column=list(original_model_lsd_df.columns), sym='C' + str(participant_id), color='C' + str(participant_id))



# mean = acc_dataframe.mean(axis=0)
# plt.plot(np.linspace(1.0, 11, 11), mean, color='C' + str(subject_index), marker="*", markersize=10)

plt.show()



