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


# argument 1 user name
# argument 2 current coming session
# argument 3 force to run all the sessions

random_state = 3
# loo_subject_name = 'Sub1_hw'
load_data_dir = '../../data/IndexPenData/IndexPenStudyData/UserStudy2Data/'
best_model_path = '......................'

sys.argv.append('User0')
sys.argv.append('1')
argv_len = sys.argv
print('Number of arguments:', argv_len, 'arguments.')
print('Argument List:', str(sys.argv))

user_name = sys.argv[1]

user_data_dir = load_data_dir+user_name

# with open(load_data_dir, 'rb') as f:
#     subjects_data_dict, subjects_label_dict, subjects_group_dict, encoder = pickle.load(f)

# load data for train and test

# train the model using best model


# save confusion matrix png, history png, loss png, [best cm, best accuracy] pickle, the best transfer model\


# lavinshtein distance for the last 5 sentences using original model
# lavinshtein distance for the last 5 sentences using the transfer model

# save lavinshtein distance for all 5 sentences
# save the prediction result
# save the ground truth

