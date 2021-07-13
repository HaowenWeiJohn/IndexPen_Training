import datetime
import glob
import os.path
import pickle

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


load_data_dir = '../../data/IndexPenData/IndexPenData2020/2020_31classes_corrupt_frame_removal_(-1000,1500)_(0,2500)_clutter_removal_(0.8)_(0.6)'

# load all data and Y
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

# create auto save dir

root_dir = 'auto_save'
os.mkdir(root_dir)

for loo_subject_name in subjects_data_dict:
    # rd and ra

    # make subject dir
    subject_dir = os.path.join(root_dir, loo_subject_name)
    train_info_dir = os.path.join(subject_dir, 'train_info')
    transfer_learning_dir = os.path.join(subject_dir, 'transfer_learning')

    X_mmw_rD_model = None
    X_mmw_rD_loo = None

    X_mmw_rA_model = None
    X_mmw_rA_loo = None

    Y_model = None
    Y_loo = None

    save_dir = 'leave_' + str(loo_subject_name)

    for subject_name in subjects_data_dict:
        if subject_name == loo_subject_name:
            X_mmw_rD_loo = subjects_data_dict[subject_name][0]
            X_mmw_rA_loo = subjects_data_dict[subject_name][1]
            Y_loo = subjects_label_dict[subject_name]
        else:
            if X_mmw_rA_model == None:
                X_mmw_rD_model = subjects_data_dict[subject_name][0]
                X_mmw_rA_model = subjects_data_dict[subject_name][1]
                Y_model = subjects_label_dict[subject_name]
            else:
                X_mmw_rD_model = np.concatenate([X_mmw_rD_model, subjects_data_dict[subject_name][0]])
                X_mmw_rA_model = np.concatenate([X_mmw_rA_model, subjects_data_dict[subject_name][1]])
                Y_model = np.concatenate([Y_model, subjects_label_dict[subject_name]])

        # train the build model data

        X_mmw_rD_model_train, X_mmw_rD_model_test, Y_model_train, Y_model_test = train_test_split(
            X_mmw_rD_model, Y_model, stratify=Y_model, test_size=0.20,
            random_state=random_state,
            shuffle=True)

        X_mmw_rA_model_train, X_mmw_rA_model_test, Y_model_train, Y_model_test = train_test_split(
            X_mmw_rA_model, Y_model, stratify=Y_model, test_size=0.20,
            random_state=random_state,
            shuffle=True)

        # build model
        model = make_simple_model(class_num=31, learning_rate=1e-3, decay=2e-6)

        # train the model with leave one out
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        # ------------------------
        model_log_csv_path = os.path.join(train_info_dir, 'model_history_log.csv')
        csv_logger = CSVLogger(model_log_csv_path, append=True)
        # ------------------------
        best_model_path = os.path.join(train_info_dir, 'best_model.h5')
        mc = ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

        training_start_time = time.time()
        # train  the model
        history = model.fit([X_mmw_rD_model_train, X_mmw_rA_model_train], Y_model_train,
                            validation_data=([X_mmw_rD_model_test, X_mmw_rA_model_test], Y_model_test),
                            epochs=20000,
                            batch_size=64, callbacks=[es, mc, csv_logger], verbose=1, shuffle=True)

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
        model_cm = plot_confusion_matrix(y_true=Y_model_test, y_pred=Y_model_pred, classes=indexpen_classes)
        plt.savefig('confusion_matrix.png')
        transfer_test_acc = accuracy_score(Y_model_test, Y_model_pred)
        print("best_accuracy_score:", transfer_test_acc)

        # save history, cm also in a pickle file.
        with open(os.path.join(train_info_dir, 'train_hist_cm.pickle'), 'wb') as f:
            pickle.dump([history, model_cm], f)

        ###################################################################################################### start transfer learning ###############################################

        # X_mmw_rD_loo # X_mmw_rA_loo # Y_loo
        '''''
            select 40 samples from each class from loo user
            20 20 split
            feed in 0, 4, 8, 12, 16, 20
        '''''
        loo_data_num = len(X_mmw_rD_loo)
        select_sample_num = 20
        transfer_data_ratio = loo_data_num / select_sample_num
        # select 20 samples from each class
        X_mmw_rD_transfer, X_mmw_rD_ignore, Y_transfer, Y_ignore = train_test_split(X_mmw_rD_loo, Y_loo,
                                                                                    stratify=Y_loo,
                                                                                    train_size=transfer_data_ratio,
                                                                                    random_state=3,
                                                                                    shuffle=True)

        X_mmw_rA_transfer, X_mmw_rA_ignore, Y_transfer, Y_ignore = train_test_split(X_mmw_rA_loo, Y_loo,
                                                                                    stratify=Y_loo,
                                                                                    train_size=transfer_data_ratio,
                                                                                    random_state=3,
                                                                                    shuffle=True)


        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=10, random_state=3)

        # feed in sample size from training
        feed_in_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
        # create cm dataframe
        best_transfer_cm_hist_dict = {}
        best_transfer_acc_hist_dict = {}

        split_round = 0

        ########################### 20 round transfer learning
        for train_ix, test_ix in rskf.split(X=X_mmw_rD_transfer, y=np.argmax(Y_transfer, axis=1)):
            split_round += 1
            print("Split Round: ", split_round)

            X_mmw_rD_transfer_train, X_mmw_rD_transfer_test = X_mmw_rD_transfer[train_ix], X_mmw_rD_transfer[test_ix]
            X_mmw_rA_transfer_train, X_mmw_rA_transfer_test = X_mmw_rA_transfer[train_ix], X_mmw_rA_transfer[test_ix]
            Y_transfer_train, Y_transfer_test = Y_transfer[train_ix], Y_transfer[test_ix]

            for feed_in_ratio in feed_in_ratios:
                print("Split Round: ", split_round, "Feed in Ratio", feed_in_ratio)
                # create transfer model
                transfer_model = make_transfer_model(pretrained_model=model,
                                                     class_num=31,
                                                     learning_rate=1e-3,
                                                     decay=4e-5,
                                                     only_last_layer_trainable=True)
                # feed in sample ratio equal to train size
                if feed_in_ratio != 1.0:
                    X_mmw_rD_transfer_feed_in, X_mmw_rD_transfer_leave_out, Y_transfer_feed_in, Y_transfer_leave_out = train_test_split(X_mmw_rD_transfer_train,
                                                                                                                                        Y_transfer_train,
                                                                                                                                        stratify=Y_transfer_train,
                                                                                                                                        train_size=feed_in_ratio,
                                                                                                                                        random_state=3,
                                                                                                                                        shuffle=True)

                    X_mmw_rA_transfer_feed_in, X_mmw_rA_transfer_leave_out, Y_transfer_feed_in, Y_transfer_leave_out = train_test_split(X_mmw_rA_transfer_train,
                                                                                                                                        Y_transfer_train,
                                                                                                                                        stratify=Y_transfer_train,
                                                                                                                                        train_size=feed_in_ratio,
                                                                                                                                        random_state=3,
                                                                                                                                        shuffle=True)
                else:
                    X_mmw_rD_transfer_feed_in = X_mmw_rD_transfer_train
                    X_mmw_rA_transfer_feed_in = X_mmw_rA_transfer_train
                    Y_transfer_feed_in = Y_transfer_train

                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

                # transfer model csv log path
                transfer_model_csv_log_path = os.path.join(transfer_learning_dir, str(split_round) + '_' + str(feed_in_ratio) + "_model_history_log.csv")
                csv_logger = CSVLogger(filename=transfer_model_csv_log_path,
                                       append=True)

                # transfer best model bath
                best_transfer_model_path = os.path.join(transfer_learning_dir, str(split_round) + '_' + str(feed_in_ratio)+ '_best_transfer_model.h5')
                mc = ModelCheckpoint(
                    filepath= best_transfer_model_path,
                    monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

                training_start_time = time.time()

                history = transfer_model.fit([X_mmw_rD_transfer_feed_in, X_mmw_rA_transfer_feed_in], Y_transfer_feed_in,
                                             validation_data=([X_mmw_rD_transfer_test, X_mmw_rA_transfer_test], Y_transfer_test),
                                             epochs=2000,
                                             batch_size=round(len(X_mmw_rD_transfer_feed_in) / 24),
                                             callbacks=[es, mc, csv_logger],
                                             verbose=1, shuffle=True)

                print("Training Duration: --- %s seconds ---" % (time.time() - training_start_time))

                best_transfer_model = tf.keras.models.load_model(best_transfer_model_path)
                Y_transfer_pred1 = best_transfer_model.predict([X_mmw_rD_transfer_test, X_mmw_rA_transfer_test])
                Y_transfer_pred_class = np.argmax(Y_transfer_pred1, axis=1)
                Y_transfer_test_class = np.argmax(Y_transfer_test, axis=1)

                _, transfer_model_cm = plot_confusion_matrix(y_true=Y_transfer_test_class, y_pred=Y_transfer_pred_class, classes=indexpen_classes,
                                              normalize=False)

                plt.savefig(os.path.join
                            (transfer_learning_dir,
                             str(split_round) + '_' + str(feed_in_ratio) + '_confusion_matrix.png')
                            )
                plt.close()

                plt.rcdefaults()
                transfer_test_acc = accuracy_score(Y_transfer_test_class, Y_transfer_pred_class)
                print(str(loo_subject_name)+ " " + str(split_round) + " " + str(feed_in_ratio) + " best_accuracy_score:", transfer_test_acc)

                # append cm to data frame
                if str(feed_in_ratio) not in best_transfer_cm_hist_dict:
                    best_transfer_cm_hist_dict[str(feed_in_ratio)] = [transfer_model_cm]
                    best_transfer_acc_hist_dict[str(feed_in_ratio)] = [transfer_test_acc]
                else:
                    best_transfer_cm_hist_dict[str(feed_in_ratio)].append(transfer_model_cm)
                    best_transfer_acc_hist_dict[str(feed_in_ratio)].append(transfer_test_acc)

        with open(os.path.join(transfer_learning_dir, 'transfer_learning_best_cm_hist_dict'), 'wb') as f:
            pickle.dump([best_transfer_cm_hist_dict, best_transfer_acc_hist_dict], f)

        

