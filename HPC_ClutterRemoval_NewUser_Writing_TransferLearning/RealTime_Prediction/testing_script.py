import datetime
import glob
import os.path
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

import sys

# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training')

from data_utils.data_config import *
from data_utils.make_model import *
from data_utils.ploting import *

random_state = 3
loo_subject_name = 'Sub1_hw'
load_data_dir = '../../data/IndexPenData/IndexPenStudyData/UserStudy1Data/7-20_all_study_data_cr_(0.8,0.8)'

# writing data dict
'''
{
rd:
ra:
ground truth:
ground truth index:
}

'''
data_dict = {}
model = None


for frame in data_dict['rd']:


