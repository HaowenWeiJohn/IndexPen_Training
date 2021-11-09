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

error_recording_dir = 'surveys'

session_ids = [1, 2, 3, 4, 5]

participant_ids = [
    1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 16, 18, 19, 20, 21
]
q = [4,5,7,9]

sessions = ['S1', 'S2', 'S3', 'S4', 'S5']

hard_gesture_df = pd.DataFrame(0, columns=indexpen_classes,
                               index=sessions)

for index, session_id in enumerate(session_ids):
    print(session_id)

    session_survey_path = os.path.join(error_recording_dir, 'IndexPen_Post_Session_' + str(session_id) + '.xlsx')
    session_survey = pd.read_excel(session_survey_path, index_col=0)

    for participant_id in participant_ids:
        hard_str = session_survey['Hard to write'].loc[['Participant_'+str(participant_id)]]
        hard_elements = str(hard_str[0]).split(",")
        print(hard_elements)
        for element in hard_elements:
            if element in indexpen_classes:
                hard_gesture_df[element].loc[[sessions[index]]] +=1