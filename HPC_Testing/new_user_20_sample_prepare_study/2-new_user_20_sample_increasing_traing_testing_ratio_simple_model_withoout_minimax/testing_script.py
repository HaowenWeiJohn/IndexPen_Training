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

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training')
from data_utils.make_model import *
from data_utils.ploting import plot_confusion_matrix

# load existing model
model = tf.keras.models.load_model('../../../model/4-simple_model_2021-06-23_00-12-01.031452.h5')
load_data_dir  = '../../../data/IndexPenData/IndexPenStudyData/NewUser20Samples/John_20_new_sample_transfer_learning_test'

# load new user data
with open(load_data_dir, 'rb') as f:
    X_dict, Y, encoder = pickle.load(f)

X_mmw_rD = X_dict[0]
X_mmw_rA = X_dict[1]

train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
# 20 samples in total increasing from 2
for train_size in train_sizes:
    # create transfer learning model
    transfer_model = make_transfer_model(pretrained_model=model,
                                         class_num=31,
                                         learning_rate=1e-4,
                                         decay=1e-6,
                                         only_last_layer_trainable=True)

    X_mmw_rD_train, X_mmw_rD_test, Y_train, Y_test = train_test_split(X_mmw_rD, Y, stratify=Y, train_size=train_size, random_state=3,
                                                                      shuffle=True)

    X_mmw_rA_train, X_mmw_rA_test, Y_train, Y_test = train_test_split(X_mmw_rA, Y, stratify=Y, train_size=train_size, random_state=3,
                                                                      shuffle=True)


    # TODO: data augmentation


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    csv_logger = CSVLogger(str(train_size)+"_model_history_log.csv", append=True)
    mc = ModelCheckpoint(
        # filepath='AutoSave/' + str(datetime.datetime.now()).replace(':', '-').replace(' ',
        filepath=str(train_size)+'_'+str(datetime.datetime.now()).replace(':', '-').replace(' ',
                                                                        '_') + '.h5',
        monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    training_start_time = time.time()

    history = transfer_model.fit([X_mmw_rD_train, X_mmw_rA_train], Y_train,
                        validation_data=([X_mmw_rD_test, X_mmw_rA_test], Y_test),
                        epochs=2000,
                        batch_size=8, callbacks=[es, mc, csv_logger], verbose=1, shuffle=True)

    print("Training Duration: --- %s seconds ---" % (time.time() - training_start_time))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(str(train_size)+'_model_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str(train_size)+'_model_accuracy.png')
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(str(train_size)+'_model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str(train_size)+'_model_loss.png')
    plt.clf()

    best_model_path = glob.glob(str(train_size)+'*.h5')[0]
    best_model = tf.keras.models.load_model(best_model_path)
    Y_pred1 = best_model.predict([X_mmw_rD_test, X_mmw_rA_test])
    Y_pred = np.argmax(Y_pred1, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    cm = plot_confusion_matrix(y_true=Y_test, y_pred=Y_pred, classes=encoder.categories_[0])
    plt.savefig(str(train_size)+'_confusion_matrix.png')
    plt.clf()

    plt.rcdefaults()
    test_acc = accuracy_score(Y_test, Y_pred)
    print("best_accuracy_score:", test_acc)

