import datetime
import glob
import os.path
import pathlib
import pickle
import shutil
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedShuffleSplit
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

import sys

# insert at 1, 0 is the script path (or '' in REPL)
# Prepare the lite model for that session. If this is session 1, we ignore it and use the generated model

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


sys.argv.append('participant_3')
sys.argv.append('session_1')


argv_len = sys.argv
print('Number of arguments:', argv_len, 'arguments.')
print('Argument List:', str(sys.argv))

participant_name = sys.argv[1]
session_name = sys.argv[2]
# offline_evaluation = sys.argv[3]

# make participant directory in participants_session_transfer_train

random_state = 3
# loo_subject_name = 'Sub1_hw'

original_model_path = indexpen_study2_original_model_path

load_data_dir = '../../data/IndexPenData/IndexPenStudyData/UserStudy2Data/'
participant_data_dir = os.path.join(load_data_dir, participant_name)


result_save_dir = '../participants_session_transfer_train'
participant_save_dir = os.path.join(result_save_dir, participant_name)
participant_session_transfer_train_dir = os.path.join(participant_save_dir, session_name)

session_index = int(session_name.split("_")[-1])

######## participant error note file


# create directory participant save dir do not exist
if not os.path.isdir(participant_save_dir):
    os.mkdir(participant_save_dir)

# recreate the session directory, re-evaluate the session directory
if os.path.isdir(participant_session_transfer_train_dir):
    shutil.rmtree(participant_session_transfer_train_dir)
os.mkdir(participant_session_transfer_train_dir)

# load all data < session index for training
# if session is 1, we copy the best model as transfer model and end

# if session_index == 1:
#     print('Session one do not need transfer learning, please analysis using best model.')
#     # original_model = tf.keras.models.load_model(original_model_path)
#     # original_model.save(os.path.join(participant_session_transfer_train_dir, 'transfer_model.h5'))
#     sys.exit(0)
error_notes = pd.read_csv(os.path.join(indexpen_study2_error_recording_dir,
                                       participant_name+
                                       '_error_recording_form.csv'), index_col=0)
train_trails = {}
trail_index = 0
for this_session_index in range(1, session_index+1):
    this_session_file_path = os.path.join(participant_data_dir, 'session_' + str(this_session_index))
    with open(this_session_file_path, 'rb') as f:
        this_session_data = pickle.load(f)
    # remove error frame using the csv file
    for error_target_trail in this_session_data:
        trail_error_list = error_notes.loc[str(error_target_trail)+'_error', :]
        for error_sample_index in range(0, len(trail_error_list)):
            if pd.isnull(trail_error_list[error_sample_index]) is False:
                    this_session_data[int(error_target_trail)][0][0][0] = \
                        np.delete(this_session_data[int(error_target_trail)][0][0][0], error_sample_index, axis=0)
                    this_session_data[int(error_target_trail)][0][0][1] = \
                        np.delete(this_session_data[int(error_target_trail)][0][0][1], error_sample_index, axis=0)

                    this_session_data[int(error_target_trail)][0] = list(this_session_data[int(error_target_trail)][0])
                    this_session_data[int(error_target_trail)][0][1] = \
                        np.delete(this_session_data[int(error_target_trail)][0][1], error_sample_index, axis=0)
                    this_session_data[int(error_target_trail)][0] = tuple(this_session_data[int(error_target_trail)][0])

    # error_samples = error_notes.loc[participant_name]['session_' + str(this_session_index)]
    # if pd.isnull(error_samples) is False:
    #     error_dict = json.loads(error_samples)
    #     for error_target_trail in error_dict:
    #         for error_sample_index in error_dict[error_target_trail]:
    #             # find the training sample index
    #             this_session_data[int(error_target_trail)][0][0][0] = \
    #                 np.delete(this_session_data[int(error_target_trail)][0][0][0], error_sample_index, axis=0)
    #             this_session_data[int(error_target_trail)][0][0][1] = \
    #                 np.delete(this_session_data[int(error_target_trail)][0][0][1], error_sample_index, axis=0)
    #
    #             this_session_data[int(error_target_trail)][0] = list(this_session_data[int(error_target_trail)][0])
    #             this_session_data[int(error_target_trail)][0][1] = \
    #                 np.delete(this_session_data[int(error_target_trail)][0][1], error_sample_index, axis=0)
    #             this_session_data[int(error_target_trail)][0] = tuple(this_session_data[int(error_target_trail)][0])





    for this_trail_index in this_session_data:
        train_trails[this_trail_index] = this_session_data[this_trail_index]
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

# TODO: check data augmentation**********


original_model = tf.keras.models.load_model(original_model_path)
transfer_model = make_transfer_model(pretrained_model=original_model,
                                     class_num=31,
                                     learning_rate=5e-4,
                                     decay=2e-5,
                                     only_last_layer_trainable=False)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# transfer model csv log path
transfer_model_csv_log_path = os.path.join(participant_session_transfer_train_dir, "transfer_model_history_log.csv")
csv_logger = CSVLogger(filename=transfer_model_csv_log_path,
                       append=True)
# save model path
transfer_model_path = os.path.join(participant_session_transfer_train_dir,
                                        'transfer_model.h5')
mc = ModelCheckpoint(
    filepath=transfer_model_path,
    monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
#
training_start_time = time.time()
#
history = transfer_model.fit([X_mmw_rD_train, X_mmw_rA_train], Y_train,
                             validation_data=(
                                 [X_mmw_rD_test, X_mmw_rA_test], Y_test),
                             epochs=1000,
                             batch_size=round(len(X_mmw_rD_train) / 32),
                             validation_batch_size=64,
                             callbacks=[es, mc, csv_logger],
                             verbose=1, shuffle=True)

print("Training Duration: --- %s seconds ---" % (time.time() - training_start_time))

################################################## trasfer model result ######################################
# save model data
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('transfer model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(participant_session_transfer_train_dir, 'transfer_model_accuracy.png'))
plt.close()
#
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('transfer model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(participant_session_transfer_train_dir, 'transfer_model_loss.png'))
plt.close()

transfer_model = tf.keras.models.load_model(transfer_model_path)
Y_transfer_pred1 = transfer_model.predict([X_mmw_rD_test, X_mmw_rA_test])
Y_transfer_pred_class = np.argmax(Y_transfer_pred1, axis=1)
Y_transfer_test_class = np.argmax(Y_test, axis=1)

_, transfer_model_cm = plot_confusion_matrix(y_true=Y_transfer_test_class, y_pred=Y_transfer_pred_class,
                                             classes=indexpen_classes,
                                             normalize=False)

plt.savefig(os.path.join(participant_session_transfer_train_dir, 'transfer_confusion_matrix.png'))
plt.close()
plt.rcdefaults()

transfer_test_acc = accuracy_score(Y_transfer_test_class, Y_transfer_pred_class)
print('best_accuracy_score: ', transfer_test_acc)

with open(os.path.join(participant_session_transfer_train_dir, 'transfer_learning_best_cm_hist_dict'), 'wb') as f:
    pickle.dump([transfer_model_cm, transfer_test_acc], f)



######################## light model converter #####################################
original_model = tf.keras.models.load_model(transfer_model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(original_model)  # path to the SavedModel directory
# converter.post_training_quantize=True
tflite_model = converter.convert()

lite_model_save_path = os.path.join(participant_session_transfer_train_dir, session_name+"_lite_models/")
tflite_models_dir = pathlib.Path(lite_model_save_path)
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir / "indexpen_model.tflite"
tflite_model_file.write_bytes(tflite_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
#
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir / "indexpen_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)


# #************************************ load current session data for evaluation: **********************************
