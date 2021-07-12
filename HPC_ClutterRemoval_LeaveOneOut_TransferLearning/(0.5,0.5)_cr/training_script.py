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

import sys

# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training')

from data_utils.data_config import *
from data_utils.make_model import *
from data_utils.ploting import *

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

for loo_subject_name in subjects_data_dict:
    # rd and ra
    X_mmw_rD_model = None
    X_mmw_rD_loo = None

    X_mmw_rA_model = None
    X_mmw_rA_loo = None

    Y_model = None
    Y_loo = None

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
            random_state=3,
            shuffle=True)

        X_mmw_rA_model_train, X_mmw_rA_model_test, Y_model_train, Y_model_test = train_test_split(
            X_mmw_rA_model, Y_model, stratify=Y_model, test_size=0.20,
            random_state=3,
            shuffle=True)

        # build model
        model = make_simple_model(class_num=31, learning_rate=1e-3, decay=2e-6)

        # train the model with leave one out
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        csv_logger = CSVLogger("auto_save/model_history_log.csv", append=True)
        mc = ModelCheckpoint(
            # filepath='AutoSave/' + str(datetime.datetime.now()).replace(':', '-').replace(' ',
            filepath=str(datetime.datetime.now()).replace(':', '-').replace(' ',
                                                                            '_') + '.h5',
            monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


        training_start_time = time.time()
        # train  the model
        history = model.fit([X_mmw_rD_model_train, X_mmw_rA_model_train], Y_model_train,
                            validation_data=([X_mmw_rD_model_test, X_mmw_rA_model_test], Y_model_test),
                            epochs=20000,
                            batch_size=64, callbacks=[es, mc, csv_logger], verbose=1, shuffle=True)

        print("Training Duration: --- %s seconds ---" % (time.time() - training_start_time))

        # save model data

        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.savefig('model_accuracy.png')
        # plt.clf()
        #
        # # summarize history for loss
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.savefig('model_loss.png')
        # plt.clf()
        #
        # best_model_path = glob.glob('./*.h5')[0]
        # best_model = tf.keras.models.load_model(best_model_path)
        # Y_pred1 = best_model.predict([X_mmw_rD_model_test, X_mmw_rA_model_test])
        # Y_pred = np.argmax(Y_pred1, axis=1)
        # Y_test = np.argmax(Y_test, axis=1)
        # cm = plot_confusion_matrix(y_true=Y_test, y_pred=Y_pred, classes=indexpen_classes)
        # plt.savefig('confusion_matrix.png')
        # test_acc = accuracy_score(Y_test, Y_pred)
        # print("best_accuracy_score:", test_acc)