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
# argument 2 the session want to evaluate
# argument 3 force to run all the sessions

random_state = 3
# loo_subject_name = 'Sub1_hw'
load_data_dir = '../../data/IndexPenData/IndexPenStudyData/UserStudy2Data/'
best_model_path = '......................'

sys.argv.append('participant_0')
sys.argv.append('session_3')
argv_len = sys.argv
print('Number of arguments:', argv_len, 'arguments.')
print('Argument List:', str(sys.argv))

participant_name = sys.argv[1]
session_name = sys.argv[2]

participant_data_dir = os.path.join(load_data_dir, participant_name)

session_index = int(session_name.split("_")[-1])

# load all data < session index for training
# if session is 1, we feed in directly
# accuracy for both trained and untrained for that session

train_trails = {}
trail_index = 0
for this_session_index in range(1, session_index):
    this_session_file_path = os.path.join(participant_data_dir, 'session_' + str(this_session_index))
    with open(this_session_file_path, 'rb') as f:
        this_session_data = pickle.load(f)
    for this_trail_index in this_session_data:
        train_trails[trail_index] = this_session_data[this_trail_index]
        trail_index += 1

################################################## training  ####################################
X_mmw_rD = []
X_mmw_rA = []
Y = []

for trail in train_trails:
    if len(X_mmw_rD) == 0:
        X_mmw_rD = train_trails[trail][0][0][0]
        X_mmw_rA = train_trails[trail][0][0][1]
        Y = train_trails[trail][0][1]
    else:
        X_mmw_rD = np.concatenate([X_mmw_rD, train_trails[trail][0][0][0]])
        X_mmw_rA = np.concatenate([X_mmw_rA, train_trails[trail][0][0][1]])
        Y = np.concatenate([Y, train_trails[trail][0][1]])

X_mmw_rD_train, X_mmw_rD_test, Y_train, Y_test = train_test_split(X_mmw_rD, Y, stratify=Y, test_size=0.20,
                                                                  random_state=random_state,
                                                                  shuffle=True)

X_mmw_rA_train, X_mmw_rA_test, Y_train, Y_test = train_test_split(X_mmw_rA, Y, stratify=Y, test_size=0.20,
                                                                  random_state=random_state,
                                                                  shuffle=True)

# TODO: check data augmentation
data_augmen


################################################## evaluation ######################################


# for session_index in  range(1, session_index+1):


# session_data = {}
# for session_data
#         with open(session_data_path, 'rb') as f:
#                 session_data = pickle.load(f)


# ../../data/IndexPenData/IndexPenStudyData/UserStudy2Data/participant_0/session_2'

# load data for train and test

# train the model using best model


# save confusion matrix png, history png, loss png, [best cm, best accuracy] pickle, the best transfer model\


# lavinshtein distance for the last 5 sentences using original model
# lavinshtein distance for the last 5 sentences using the transfer model

# save lavinshtein distance for all 5 sentences
# save the prediction result
# save the ground truth
