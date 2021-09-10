import datetime
import glob
import os.path
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedShuffleSplit
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

import sys

# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training')

from data_utils.data_config import *
from data_utils.make_model import *
from data_utils.ploting import *
from data_utils.data_general_utils import *

random_state = 3
# loo_subject_name = 'Sub1_hw'
load_data_dir = '../data/IndexPenData/IndexPenStudyData/UserStudy1Data/8-13_5User_cr_(0.8,0.8)'

# load all data and Y
with open(load_data_dir, 'rb') as f:
    subjects_data_dict, subjects_label_dict, subjects_group_dict, encoder = pickle.load(f)


first_A = subjects_data_dict['Sub1_hw'][0][15]
first_B = subjects_data_dict['Sub1_hw'][0][7]
first_C = subjects_data_dict['Sub1_hw'][0][51]
known_A = subjects_data_dict['Sub1_hw'][0][25]
known_B = subjects_data_dict['Sub1_hw'][0][28]
known_C = subjects_data_dict['Sub1_hw'][0][57]

start = time.time()
distance_A_A = dtw_rd(first_A, known_A, filename='distance_A_A')
distance_A_B = dtw_rd(first_B, known_A, filename='distance_A_B')
distance_A_C = dtw_rd(first_A, known_C, filename='distance_A_C')
distance_B_B = dtw_rd(first_B, known_B, filename='distance_B_B')
distance_B_C = dtw_rd(first_B, known_C, filename='distance_B_C')
distance_C_C = dtw_rd(first_C, known_C, filename='distance_C_C')
print(time.time()-start)
