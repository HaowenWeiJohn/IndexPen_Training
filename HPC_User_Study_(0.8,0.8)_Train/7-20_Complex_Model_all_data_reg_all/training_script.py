import datetime
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

import sys

# insert at 1, 0 is the script path (or '' in REPL)


sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training')

from data_utils.data_config import *
from data_utils.make_model import *
from data_utils.ploting import plot_confusion_matrix

load_data_dir = '../../data/IndexPenData/IndexPenStudyData/UserStudy1Data/7-20_all_study_data_cr_(0.8,0.8)'
# load all data and Y
subject_name = 'Sub2_zs'

with open(load_data_dir, 'rb') as f:
    subjects_data_dict, subjects_label_dict, encoder = pickle.load(f)

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
X_mmw_rD = []
X_mmw_rA = []
Y = []
# load data
subject_names = ['Sub1_hw', 'Sub2_zs', 'Sub3_ag', 'Sub4_xz']
for subject_name in subject_names:
    if len(X_mmw_rD) == 0:
        X_mmw_rD = subjects_data_dict[subject_name][0]
        X_mmw_rA = subjects_data_dict[subject_name][1]
        Y = subjects_label_dict[subject_name]
    else:
        X_mmw_rD = np.concatenate([X_mmw_rD, subjects_data_dict[subject_name][0]])
        X_mmw_rA = np.concatenate([X_mmw_rA, subjects_data_dict[subject_name][1]])
        Y = np.concatenate([Y, subjects_label_dict[subject_name]])

# print(np.min(X_mmw_rD))
# print(np.max(X_mmw_rD))
# print(np.min(X_mmw_rA))
# print(np.max(X_mmw_rA))
#
# rD_min = -1000
# rD_max = 1500
# rA_min = 0
# rA_max = 2500

# unit minimax:
# for index in range(0, len(X_mmw_rD)):
#     X_mmw_rD[index] = (X_mmw_rD[index] - np.min(X_mmw_rD[index])) / (np.max(X_mmw_rD[index]) - np.min(X_mmw_rD[index]))
#
# for index in range(0, len(X_mmw_rA)):
#     X_mmw_rA[index] = (X_mmw_rA[index] - np.min(X_mmw_rA[index])) / (np.max(X_mmw_rA[index]) - np.min(X_mmw_rA[index]))
# X_mmw_rD = (X_mmw_rD - rD_min) / (rD_max - rD_min)
# X_mmw_rA = (X_mmw_rA - rA_min) / (rA_max - rA_min)


X_mmw_rD_train, X_mmw_rD_test, Y_train, Y_test = train_test_split(X_mmw_rD, Y, stratify=Y, test_size=0.20,
                                                                  random_state=3,
                                                                  shuffle=True)

X_mmw_rA_train, X_mmw_rA_test, Y_train, Y_test = train_test_split(X_mmw_rA, Y, stratify=Y, test_size=0.20,
                                                                  random_state=3,
                                                                  shuffle=True)

model = make_complex_model(class_num=31, learning_rate=1e-3, decay=2e-6,
                           rd_kernel_size1=(3, 3), rd_kernel_size2=(3, 3),
                           ra_kernel_size1=(3, 3), ra_kernel_size2=(3, 3),
                           )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=250)
csv_logger = CSVLogger("model_history_log.csv", append=True)
mc = ModelCheckpoint(
    # filepath='AutoSave/' + str(datetime.datetime.now()).replace(':', '-').replace(' ',
    filepath=str(datetime.datetime.now()).replace(':', '-').replace(' ',
                                                                    '_') + '.h5',
    monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

training_start_time = time.time()

history = model.fit([X_mmw_rD_train, X_mmw_rA_train], Y_train,
                    validation_data=([X_mmw_rD_test, X_mmw_rA_test], Y_test),
                    epochs=20000,
                    batch_size=64, callbacks=[es, mc, csv_logger], verbose=1, shuffle=True)

print("Training Duration: --- %s seconds ---" % (time.time() - training_start_time))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_accuracy.png')
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_loss.png')
plt.clf()

best_model_path = glob.glob('./*.h5')[0]
best_model = tf.keras.models.load_model(best_model_path)
Y_pred1 = best_model.predict([X_mmw_rD_test, X_mmw_rA_test])
Y_pred = np.argmax(Y_pred1, axis=1)
Y_test = np.argmax(Y_test, axis=1)
cm = plot_confusion_matrix(y_true=Y_test, y_pred=Y_pred, classes=indexpen_classes)
plt.savefig('confusion_matrix.png')
test_acc = accuracy_score(Y_test, Y_pred)
print("best_accuracy_score:", test_acc)
