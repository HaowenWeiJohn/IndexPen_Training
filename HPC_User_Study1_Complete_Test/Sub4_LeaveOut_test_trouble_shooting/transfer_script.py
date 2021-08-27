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
loo_subject_name = 'Sub4_xz'
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

train_info_dir = os.path.join(save_dir, 'train_info')
transfer_info_dir = os.path.join(save_dir, 'transfer_info')

if os.path.isdir(transfer_info_dir):
    # del dir
    shutil.rmtree(transfer_info_dir)
os.mkdir(transfer_info_dir)

X_mmw_rD_loo = subjects_data_dict[loo_subject_name][0]
X_mmw_rA_loo = subjects_data_dict[loo_subject_name][1]
Y_loo = subjects_label_dict[loo_subject_name]

del subjects_data_dict

# load leave one out model
best_model_path = os.path.join(train_info_dir, 'best_model.h5')
best_model = tf.keras.models.load_model(best_model_path)

# 200 samples for each class
# create cm dataframe
best_transfer_cm_hist_dict = {}
best_transfer_acc_hist_dict = {}

train_test_split_indexes = StratifiedShuffleSplit(n_splits=10, test_size=0.9, random_state=3). \
    split(X=X_mmw_rD_loo, y=np.argmax(Y_loo, axis=1))

feed_in_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

split_round = 0

for train_ix, test_ix in train_test_split_indexes:
    split_round += 1

    X_mmw_rD_transfer_train, X_mmw_rD_transfer_test = X_mmw_rD_loo[train_ix], X_mmw_rD_loo[test_ix]
    X_mmw_rA_transfer_train, X_mmw_rA_transfer_test = X_mmw_rA_loo[train_ix], X_mmw_rA_loo[test_ix]
    Y_transfer_train, Y_transfer_test = Y_loo[train_ix], Y_loo[test_ix]

    for feed_in_ratio in feed_in_ratios:
        if feed_in_ratio <= 0.5:
            batch_size = 8
        else:
            batch_size = 16

        if feed_in_ratio != 0.0:
            # create transfer model
            best_model = tf.keras.models.clone_model(best_model)
            transfer_model = make_transfer_model(pretrained_model=best_model,
                                                 class_num=31,
                                                 learning_rate=5e-4,
                                                 decay=2e-5,
                                                 only_last_layer_trainable=True)
            # feed in sample ratio equal to train size
            if feed_in_ratio != 1.0:
                X_mmw_rD_transfer_feed_in, X_mmw_rD_transfer_leave_out, Y_transfer_feed_in, Y_transfer_leave_out = train_test_split(
                    X_mmw_rD_transfer_train,
                    Y_transfer_train,
                    stratify=Y_transfer_train,
                    train_size=feed_in_ratio,
                    random_state=3,
                    shuffle=True)

                X_mmw_rA_transfer_feed_in, X_mmw_rA_transfer_leave_out, Y_transfer_feed_in, Y_transfer_leave_out = train_test_split(
                    X_mmw_rA_transfer_train,
                    Y_transfer_train,
                    stratify=Y_transfer_train,
                    train_size=feed_in_ratio,
                    random_state=3,
                    shuffle=True)
            else:
                X_mmw_rD_transfer_feed_in = X_mmw_rD_transfer_train
                X_mmw_rA_transfer_feed_in = X_mmw_rA_transfer_train
                Y_transfer_feed_in = Y_transfer_train

            print('  ')
            print("Split Round: ", split_round, "Feed in Ratio", feed_in_ratio)
            print('Train Sample Num: ', len(X_mmw_rD_transfer_train))
            print('Feed in Sample Num: ', len(X_mmw_rD_transfer_feed_in))
            print('Test Sample Num: ', len(X_mmw_rD_transfer_test))
            print('Batch Size: ', round(len(X_mmw_rD_transfer_feed_in) / 32))


            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
            # transfer model csv log path
            transfer_model_csv_log_path = os.path.join(transfer_info_dir,
                                                       str(loo_subject_name) + '_' + str(split_round) + '_' + str(
                                                           feed_in_ratio) + "_model_history_log.csv")
            csv_logger = CSVLogger(filename=transfer_model_csv_log_path,
                                   append=True)

            best_transfer_model_path = os.path.join(transfer_info_dir,
                                                    str(loo_subject_name) + '_' + str(split_round) + '_' + str(
                                                        feed_in_ratio) + '_best_transfer_model.h5')
            mc = ModelCheckpoint(
                filepath=best_transfer_model_path,
                monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

            training_start_time = time.time()

            history = transfer_model.fit([X_mmw_rD_transfer_feed_in, X_mmw_rA_transfer_feed_in], Y_transfer_feed_in,
                                         validation_data=(
                                             [X_mmw_rD_transfer_test, X_mmw_rA_transfer_test], Y_transfer_test),
                                         epochs=1000,
                                         batch_size=round(len(X_mmw_rD_transfer_feed_in)/32),
                                         validation_batch_size=256,
                                         callbacks=[es, mc, csv_logger],
                                         verbose=1, shuffle=True)

            print("Training Duration: --- %s seconds ---" % (time.time() - training_start_time))

            # save model data
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('transfer model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(os.path.join(transfer_info_dir, str(loo_subject_name) + "_" + str(split_round) + "_" + str(
                feed_in_ratio) + '_model_accuracy.png'))
            plt.close()
            #
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('transfer model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(os.path.join(transfer_info_dir, str(loo_subject_name) + "_" + str(split_round) + "_" + str(
                feed_in_ratio) + '_model_loss.png'))
            plt.close()

            best_transfer_model = tf.keras.models.load_model(best_transfer_model_path)
            Y_transfer_pred1 = best_transfer_model.predict([X_mmw_rD_transfer_test, X_mmw_rA_transfer_test])
            Y_transfer_pred_class = np.argmax(Y_transfer_pred1, axis=1)
            Y_transfer_test_class = np.argmax(Y_transfer_test, axis=1)

            _, transfer_model_cm = plot_confusion_matrix(y_true=Y_transfer_test_class, y_pred=Y_transfer_pred_class,
                                                         classes=indexpen_classes,
                                                         normalize=False)

            plt.savefig(os.path.join
                        (transfer_info_dir,
                         str(loo_subject_name) + "_" + str(split_round) + "_" + str(
                             feed_in_ratio) + '_confusion_matrix.png')
                        )
            plt.close()

            plt.rcdefaults()
            transfer_test_acc = accuracy_score(Y_transfer_test_class, Y_transfer_pred_class)
            print(str(loo_subject_name) + " " + str(split_round) + " " + str(feed_in_ratio) + " best_accuracy_score:",
                  transfer_test_acc)

            # append cm to data frame
            if str(feed_in_ratio) not in best_transfer_cm_hist_dict:
                best_transfer_cm_hist_dict[str(feed_in_ratio)] = [transfer_model_cm]
                best_transfer_acc_hist_dict[str(feed_in_ratio)] = [transfer_test_acc]
            else:
                best_transfer_cm_hist_dict[str(feed_in_ratio)].append(transfer_model_cm)
                best_transfer_acc_hist_dict[str(feed_in_ratio)].append(transfer_test_acc)

        else:
            best_transfer_model = tf.keras.models.clone_model(best_model)
            Y_transfer_pred1 = best_transfer_model.predict([X_mmw_rD_transfer_test, X_mmw_rA_transfer_test])
            Y_transfer_pred_class = np.argmax(Y_transfer_pred1, axis=1)
            Y_transfer_test_class = np.argmax(Y_transfer_test, axis=1)

            _, transfer_model_cm = plot_confusion_matrix(y_true=Y_transfer_test_class, y_pred=Y_transfer_pred_class,
                                                         classes=indexpen_classes,
                                                         normalize=False)

            plt.savefig(os.path.join
                        (transfer_info_dir,
                         str(loo_subject_name) + "_" + str(split_round) + "_" + str(
                             feed_in_ratio) + '_confusion_matrix.png')
                        )
            plt.close()

            plt.rcdefaults()
            transfer_test_acc = accuracy_score(Y_transfer_test_class, Y_transfer_pred_class)
            print(str(loo_subject_name) + " " + str(split_round) + " " + str(feed_in_ratio) + " best_accuracy_score:",
                  transfer_test_acc)

            # append cm to data frame
            if str(feed_in_ratio) not in best_transfer_cm_hist_dict:
                best_transfer_cm_hist_dict[str(feed_in_ratio)] = [transfer_model_cm]
                best_transfer_acc_hist_dict[str(feed_in_ratio)] = [transfer_test_acc]
            else:
                best_transfer_cm_hist_dict[str(feed_in_ratio)].append(transfer_model_cm)
                best_transfer_acc_hist_dict[str(feed_in_ratio)].append(transfer_test_acc)

with open(os.path.join(transfer_info_dir, 'transfer_learning_best_cm_hist_dict'), 'wb') as f:
    pickle.dump([best_transfer_cm_hist_dict, best_transfer_acc_hist_dict], f)

# for feed_in_ratio in feed_in_ratios:
#     if feed_in_ratio == 0:
#         # direct test with 200 samples
#         best_transfer_model = best_model
#         Y_transfer_pred1 = best_transfer_model.predict([X_mmw_rD_loo, X_mmw_rA_loo])
#         Y_transfer_pred_class = np.argmax(Y_transfer_pred1, axis=1)
#         Y_transfer_test_class = np.argmax(Y_loo, axis=1)
#
#         _, transfer_model_cm = plot_confusion_matrix(y_true=Y_transfer_test_class, y_pred=Y_transfer_pred_class, classes=indexpen_classes,
#                                       normalize=False)
#
#         plt.savefig(os.path.join
#                     (transfer_info_dir,
#                      str(feed_in_ratio) + '_confusion_matrix.png')
#                     )
#         plt.close()
#
#         plt.rcdefaults()
#
#         transfer_test_acc = accuracy_score(Y_transfer_test_class, Y_transfer_pred_class)
#         print(str(loo_subject_name) + " " + str(feed_in_ratio) + " best_accuracy_score:", transfer_test_acc)
#
