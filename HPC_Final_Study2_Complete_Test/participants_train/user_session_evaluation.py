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



# load current session data

sys.argv.append('participant_0')
sys.argv.append('session_2')
# sys.argv.append('1')

argv_len = sys.argv
print('Number of arguments:', argv_len, 'arguments.')
print('Argument List:', str(sys.argv))

participant_name = sys.argv[1]
session_name = sys.argv[2]
# offline_evaluation = sys.argv[3]


random_state = 3

best_model_path = '../../HPC_Final_Study1_Complete_Test/5User_Ultimate_Model/auto_save/train_info/best_model.h5'

load_data_dir = '../../data/IndexPenData/IndexPenStudyData/UserStudy2Data/'

transfer_result_save_dir = '../participants_session_transfer_train'

evaluation_result_save_dir = '../participant_session_transfer_evaluation'

participant_data_dir = os.path.join(load_data_dir, participant_name)

participant_session_transfer_train_dir = os.path.join(transfer_result_save_dir, participant_name)


session_index = int(session_name.split("_")[-1])

# if not os.path.isdir(participant_session_transfer_train_dir):
#     print('Please run the transfer learning model first')
#     sys.exit(0)

# create participant file



best_transfer_model_path = os.path.join(participant_session_transfer_train_dir, 'best_transfer_model.h5')

best_transfer_model = tf.keras.models.load_model(best_transfer_model_path)

with open(os.path.join(participant_data_dir, session_name), 'rb') as f:
    evaluate_session_data = pickle.load(f)


X_mmw_rD_evaluate = []
X_mmw_rA_evaluate = []
Y_evaluate = []

for evaluate_trail in evaluate_session_data:
    if len(X_mmw_rD_evaluate) == 0:
        X_mmw_rD_evaluate = evaluate_session_data[evaluate_trail][0][0][0]
        X_mmw_rA_evaluate = evaluate_session_data[evaluate_trail][0][0][1]
        Y_evaluate = evaluate_session_data[evaluate_trail][0][1]
    else:
        X_mmw_rD_evaluate = np.concatenate([X_mmw_rD_evaluate, evaluate_session_data[evaluate_trail][0][0][0]])
        X_mmw_rA_evaluate = np.concatenate([X_mmw_rA_evaluate, evaluate_session_data[evaluate_trail][0][0][1]])
        Y_evaluate = np.concatenate([Y_evaluate, evaluate_session_data[evaluate_trail][0][1]])

evaluate_model = tf.keras.models.load_model(best_transfer_model_path)
Y_evaluate_pred1 = evaluate_model.predict([X_mmw_rD_evaluate, X_mmw_rA_evaluate])
Y_evaluate_pred_class = np.argmax(Y_evaluate_pred1, axis=1)
Y_evaluate_test_class = np.argmax(Y_evaluate, axis=1)

_, evaluate_model_cm = plot_confusion_matrix(y_true=Y_evaluate_test_class, y_pred=Y_evaluate_pred_class,
                                             classes=indexpen_classes,
                                             normalize=False)

plt.savefig(os.path.join(participant_session_transfer_train_dir, 'evaluate_session_confusion_matrix.png'))
plt.close()
plt.rcdefaults()

evaluate_session_test_acc = accuracy_score(Y_evaluate_test_class, Y_evaluate_pred_class)
print("best_evaluate_session_accuracy_score:",
      evaluate_session_test_acc)

with open(os.path.join(participant_session_transfer_train_dir, 'evaluate_session_best_cm_hist_dict'), 'wb') as f:
    pickle.dump([evaluate_model_cm, evaluate_session_test_acc], f)

# run sentences through realtime debouncer algorithm
