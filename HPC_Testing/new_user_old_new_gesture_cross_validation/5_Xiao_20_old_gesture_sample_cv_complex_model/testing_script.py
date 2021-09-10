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
from data_utils.data_general_utils import *
from data_utils.ploting import *
from data_utils.data_config import *

# load existing model ## Simple model without minimax
model = tf.keras.models.load_model(
    '../../../model/3-Alex_John_Leo_complex_model_without_minimax_2021-06-23_00-07-57.990711.h5')
load_data_dir = '../../../data/IndexPenData/IndexPenStudyData/NewUser20Samples/Xiao/Xiao_20_new_gesture_sample_transfer_learning_test'

# cross_validation

with open(load_data_dir, 'rb') as f:
    X_dict, Y, encoder = pickle.load(f)
X_mmw_rD = X_dict[0]
X_mmw_rA = X_dict[1]

# 20 round in total
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=10, random_state=3)

# feed in sample size from training
feed_in_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

# create cm dataframe
best_cm_hist_dict = {}
best_acc_hist_dict = {}


split_round = 0
for train_ix, test_ix in rskf.split(X=X_mmw_rD, y=np.argmax(Y, axis=1)):
    split_round += 1
    print("Split Round: ", split_round)

    X_mmw_rD_train, X_mmw_rD_test = X_mmw_rD[train_ix], X_mmw_rD[test_ix]
    X_mmw_rA_train, X_mmw_rA_test = X_mmw_rA[train_ix], X_mmw_rA[test_ix]
    Y_train, Y_test = Y[train_ix], Y[test_ix]

    for feed_in_ratio in feed_in_ratios:
        print("Split Round: ", split_round, "Feed in Ratio", feed_in_ratio)
        # create transfer model
        transfer_model = make_transfer_model(pretrained_model=model,
                                             class_num=31,
                                             learning_rate=1e-4,
                                             decay=6e-2,
                                             only_last_layer_trainable=True)
        # feed in sample ratio equal to train size
        if feed_in_ratio != 1.0:
            X_mmw_rD_feed_in, X_mmw_rD_leave_out, Y_feed_in, Y_leave_out = train_test_split(X_mmw_rD_train, Y_train,
                                                                                            stratify=Y_train,
                                                                                            train_size=feed_in_ratio,
                                                                                            random_state=3,
                                                                                            shuffle=True)

            X_mmw_rA_feed_in, X_mmw_rA_leave_out, Y_feed_in, Y_leave_out = train_test_split(X_mmw_rA_train, Y_train,
                                                                                            stratify=Y_train,
                                                                                            train_size=feed_in_ratio,
                                                                                            random_state=3,
                                                                                            shuffle=True)
        else:
            X_mmw_rD_feed_in = X_mmw_rD_train
            X_mmw_rA_feed_in = X_mmw_rA_train
            Y_feed_in = Y_train

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        csv_logger = CSVLogger(str(split_round) + '_' + str(feed_in_ratio) + "_model_history_log.csv", append=True)
        mc = ModelCheckpoint(
            # filepath='AutoSave/' + str(datetime.datetime.now()).replace(':', '-').replace(' ',
            filepath=str(split_round) + '_' + str(feed_in_ratio) + '_' + str(datetime.datetime.now()).replace(':',
                                                                                                              '-').replace(
                ' ',
                '_') + '.h5',
            monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

        training_start_time = time.time()

        history = transfer_model.fit([X_mmw_rD_feed_in, X_mmw_rA_feed_in], Y_feed_in,
                                     validation_data=([X_mmw_rD_test, X_mmw_rA_test], Y_test),
                                     epochs=2000,
                                     batch_size=round(len(X_mmw_rD_feed_in) / 24), callbacks=[es, mc, csv_logger],
                                     verbose=1, shuffle=True)

        print("Training Duration: --- %s seconds ---" % (time.time() - training_start_time))

        best_model_path = glob.glob(str(split_round) + '_' + str(feed_in_ratio) + '*.h5')[0]
        best_model = tf.keras.models.load_model(best_model_path)
        Y_pred1 = best_model.predict([X_mmw_rD_test, X_mmw_rA_test])
        Y_pred_class = np.argmax(Y_pred1, axis=1)
        Y_test_class = np.argmax(Y_test, axis=1)

        _, cm = plot_confusion_matrix(y_true=Y_test_class, y_pred=Y_pred_class, classes=indexpen_classes,
                                      normalize=False)
        plt.savefig(str(split_round) + '_' + str(feed_in_ratio) + '_confusion_matrix.png')
        plt.close()

        plt.rcdefaults()
        test_acc = accuracy_score(Y_test_class, Y_pred_class)
        print(str(split_round) + " " + str(feed_in_ratio) + " best_accuracy_score:", test_acc)

        # append cm to data frame
        if str(feed_in_ratio) not in best_cm_hist_dict:
            best_cm_hist_dict[str(feed_in_ratio)] = [cm]
            best_acc_hist_dict[str(feed_in_ratio)] = [test_acc]
        else:
            best_cm_hist_dict[str(feed_in_ratio)].append(cm)
            best_acc_hist_dict[str(feed_in_ratio)].append(test_acc)

with open('Xiao_old_gesture_cv_best_cm_acc_hist_dict', 'wb') as f:
    pickle.dump([best_cm_hist_dict, best_acc_hist_dict], f)
