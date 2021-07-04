from sklearn.model_selection import train_test_split
import datetime
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training')
from data_utils.make_model import *
from data_utils.data_preprocessing import *
from data_utils.ploting import *
from data_utils.data_config import *

# load existing model ## Simple model without minimax

load_data_dir = '../../data/IndexPenData/IndexPenStudyData/NewUser20Samples/John_20_new_gesture_sample_transfer_learning_test'
# load_data_dir = 'C:/Users/Haowe/OneDrive/Desktop/IndexPenData2020/2020_31classes_corrupt_frame_removal_(-1000,1500)_(0,2500)'
with open(load_data_dir, 'rb') as f:
    X_dict, Y, encoder = pickle.load(f)
X_mmw_rD_test = X_dict[0]
X_mmw_rA_test = X_dict[1]

# load_data_dir = '/work/hwei/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training/data/IndexPenData/IndexPenData2020/2020_31classes_corrupt_frame_removal_(-1000,1500)_(0,2500)'
# with open(load_data_dir, 'rb') as f:
#     X_mmw_rD_test, X_mmw_rA_test, Y, encoder = pickle.load(f)

best_model_path = glob.glob(
    '../../model/4-Alex_John_Leo_simple_model_without_minimax_2021-06-23_00-12-01.031452.h5')[0]
best_model = tf.keras.models.load_model(best_model_path)
Y_pred1 = best_model.predict([X_mmw_rD_test, X_mmw_rA_test])
Y_pred_class = np.argmax(Y_pred1, axis=1)
Y_test_class = np.argmax(Y, axis=1)

_, cm = plot_confusion_matrix(y_true=Y_test_class, y_pred=Y_pred_class, classes=indexpen_classes,
                              normalize=False)
plt.savefig('confusion_matrix_6.png')
test_acc = accuracy_score(Y_test_class, Y_pred_class)
print("best_accuracy_score:", test_acc)