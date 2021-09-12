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

random_state = 3
# loo_subject_name = 'Sub1_hw'
load_data_dir = '../../data/IndexPenData/IndexPenStudyData/UserStudy1Data/8-13_5User_cr_(0.8,0.8)'


'''
subjects_data_dict: {
sub1: { rd:
        ra:
        }
sub2: { rd:
        ra:
        }
sub3: { rd:
        ra:
        }
....................
                    }
'''

save_dir = 'auto_save'
if os.path.isdir(save_dir):
    # del dir
    shutil.rmtree(save_dir)

os.mkdir(save_dir)

train_info_dir = os.path.join(save_dir, 'train_info')

os.mkdir(train_info_dir)

model = make_simple_model_reg_archive(class_num=31, learning_rate=1e-3, decay=1e-5, rd_kernel_size=(3, 3), ra_kernel_size=(3, 3),
                                      cv_reg=2e-5)

model.save('fresh_model.h5')

# train the model with leave one out
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)
# ------------------------
model_log_csv_path = os.path.join(train_info_dir, 'model_history_log.csv')
csv_logger = CSVLogger(model_log_csv_path, append=True)

best_model_path = os.path.join(train_info_dir, 'best_model.h5')

mc = ModelCheckpoint(
    filepath=best_model_path,
    monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

training_start_time = time.time()


print("Training Duration: --- %s seconds ---" % (time.time() - training_start_time))

