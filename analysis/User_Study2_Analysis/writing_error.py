# analysis of writing errors for user

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
from analysis.User_Study2_Analysis.info_extraction_utils import extract_participant_info, getCount

# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training')

from data_utils.data_config import *
from data_utils.make_model import *
from data_utils.ploting import *
from data_utils.data_config import *
from data_utils.prediction_utils import *

error_recording_dir = '../../HPC_Final_Study2_Complete_Test_Class_Balance_run_all/' \
                      'participants_data_analysis/' \
                      'participant_error_recording'

participant_ids = [
    1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 16, 18, 19, 20, 21
]

resign_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

error_num = 0
gesture_num = 0

for index, participant_id in enumerate(participant_ids):
    print(participant_id)
    error_recording_file_name = 'participant_'+str(participant_id)+'_error_recording_form.csv'
    error_recording_file = os.path.join(error_recording_dir, error_recording_file_name)
    # print(error_recording_file)

    error_notes = pd.read_csv(error_recording_file, index_col=0)
    for sentence_index in range(20, 31):
        sentence = error_notes.loc[str(sentence_index)]
        gesture_num += getCount(sentence, len(sentence), num1='Nois', num2='Nois')
        for error_index, error in enumerate(error_notes.loc[str(sentence_index)+'_error']):
            if pd.isnull(error) is False:
                error_sample = error_notes.loc[str(sentence_index)][error_index]
                error_num +=1

print(gesture_num)
print(error_num)