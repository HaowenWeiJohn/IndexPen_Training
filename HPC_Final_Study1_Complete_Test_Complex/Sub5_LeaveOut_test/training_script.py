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
loo_subject_name = 'Sub5_dt'
load_data_dir = '../../data/IndexPenData/IndexPenStudyData/UserStudy1Data/8-13_5User_cr_(0.8,0.8)'

# load all data and Y
with open(load_data_dir, 'rb') as f:
    subjects_data_dict, subjects_label_dict, subjects_group_dict, encoder = pickle.load(f)

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

# extract
X_mmw_rD_model = []
X_mmw_rD_loo = []

X_mmw_rA_model = []
X_mmw_rA_loo = []

Y_model = []
Y_loo = []

group_model = []
group_loo = []

for subject_name in subjects_data_dict:
    if subject_name == loo_subject_name:
        X_mmw_rD_loo = subjects_data_dict[subject_name][0]
        X_mmw_rA_loo = subjects_data_dict[subject_name][1]
        Y_loo = subjects_label_dict[subject_name]
        group_loo = subjects_group_dict[subject_name]
    else:
        if len(X_mmw_rA_model) == 0:
            X_mmw_rD_model = subjects_data_dict[subject_name][0]
            X_mmw_rA_model = subjects_data_dict[subject_name][1]
            Y_model = subjects_label_dict[subject_name]
            group_model = subjects_group_dict[subject_name]
        elif subject_name != 'Sub5_someone':
            X_mmw_rD_model = np.concatenate([X_mmw_rD_model, subjects_data_dict[subject_name][0]])
            X_mmw_rA_model = np.concatenate([X_mmw_rA_model, subjects_data_dict[subject_name][1]])
            Y_model = np.concatenate([Y_model, subjects_label_dict[subject_name]])
            group_model = np.concatenate([group_model, subjects_group_dict[subject_name]])

del subjects_data_dict
del subjects_label_dict

train_indexes, test_indexes = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=3). \
    split(X=X_mmw_rD_model, y=np.argmax(Y_model, axis=1), groups=group_model))

X_mmw_rD_model_train = X_mmw_rD_model[train_indexes]
X_mmw_rD_model_test = X_mmw_rD_model[test_indexes]

X_mmw_rA_model_train = X_mmw_rA_model[train_indexes]
X_mmw_rA_model_test = X_mmw_rA_model[test_indexes]

Y_model_train = Y_model[train_indexes]
Y_model_test = Y_model[test_indexes]

del X_mmw_rD_model
del X_mmw_rA_model
del Y_model

# X_mmw_rD_model_train, X_mmw_rD_model_test, Y_model_train, Y_model_test = train_test_split(
#     X_mmw_rD_model, Y_model, stratify=Y_model, test_size=0.20,
#     random_state=random_state,
#     shuffle=True)
#
# X_mmw_rA_model_train, X_mmw_rA_model_test, Y_model_train, Y_model_test = train_test_split(
#     X_mmw_rA_model, Y_model, stratify=Y_model, test_size=0.20,
#     random_state=random_state,
#     shuffle=True)

model = make_complex_model(class_num=31, learning_rate=1e-3, decay=1e-5,
                           rd_kernel_size1=(3, 3), rd_kernel_size2=(3, 3),
                           ra_kernel_size1=(3, 3), ra_kernel_size2=(3, 3),
                           cv_reg=2e-5,
                           )

# train the model with leave one out
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
# ------------------------
model_log_csv_path = os.path.join(train_info_dir, 'model_history_log.csv')
csv_logger = CSVLogger(model_log_csv_path, append=True)

best_model_path = os.path.join(train_info_dir, 'best_model.h5')

mc = ModelCheckpoint(
    filepath=best_model_path,
    monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

training_start_time = time.time()

# train  the model
history = model.fit([X_mmw_rD_model_train, X_mmw_rA_model_train], Y_model_train,
                    validation_data=([X_mmw_rD_model_test, X_mmw_rA_model_test], Y_model_test),
                    epochs=2000,
                    batch_size=128, callbacks=[es, mc, csv_logger], verbose=1, shuffle=True)

print("Training Duration: --- %s seconds ---" % (time.time() - training_start_time))

# save model data
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(train_info_dir, 'model_accuracy.png'))
plt.clf()
#
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(train_info_dir, 'model_loss.png'))
plt.clf()

best_model = tf.keras.models.load_model(best_model_path)
Y_model_pred1 = best_model.predict([X_mmw_rD_model_test, X_mmw_rA_model_test])
Y_model_pred = np.argmax(Y_model_pred1, axis=1)
Y_model_test = np.argmax(Y_model_test, axis=1)
_, model_cm = plot_confusion_matrix(y_true=Y_model_test, y_pred=Y_model_pred, classes=indexpen_classes)
plt.savefig(os.path.join(train_info_dir, 'confusion_matrix.png'))
transfer_test_acc = accuracy_score(Y_model_test, Y_model_pred)
print("best_accuracy_score:", transfer_test_acc)

# save history, cm also in a pickle file.
with open(os.path.join(train_info_dir, 'train_hist_cm.pickle'), 'wb') as f:
    pickle.dump([history.history, model_cm, Y_model_test, Y_model_pred], f)

# reset plt config
plt.rcdefaults()
