import datetime
import glob
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import os
import sys

# insert at 1, 0 is the script path (or '' in REPL)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(1, '/work/hwei/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training')

from data_utils.data_config import *
from data_utils.make_model import *
from data_utils.ploting import plot_confusion_matrix

load_data_dir = '../../data/IndexPenData/IndexPenStudyData/UserStudy1Data/8-2_4User_cr_(0.8,0.8)'
# load all data and Y


# with open(load_data_dir, 'rb') as f:
#     subjects_data_dict, subjects_label_dict, encoder = pickle.load(f)
#
# '''
# subjects_data_dict: {
# sub1: { rd:
#         ra:
#         }
# sub2: { rd:
#         ra:
#         }
# sub3: { rd:
#         ra:
#         }
# ....................
#                     }
# '''
# X_mmw_rD = []
# X_mmw_rA = []
# Y = []
# # load data
#
# for subject_name in subjects_data_dict:
#     if len(X_mmw_rD) == 0:
#         X_mmw_rD = subjects_data_dict[subject_name][0]
#         X_mmw_rA = subjects_data_dict[subject_name][1]
#         Y = subjects_label_dict[subject_name]
#     else:
#         X_mmw_rD = np.concatenate([X_mmw_rD, subjects_data_dict[subject_name][0]])
#         X_mmw_rA = np.concatenate([X_mmw_rA, subjects_data_dict[subject_name][1]])
#         Y = np.concatenate([Y, subjects_label_dict[subject_name]])
#
# del subjects_data_dict
# del subjects_label_dict
#
# X_mmw_rD_train, X_mmw_rD_test, Y_train, Y_test = train_test_split(X_mmw_rD, Y, stratify=Y, test_size=0.20,
#                                                                   random_state=3,
#                                                                   shuffle=True)
#
# del X_mmw_rD
#
# X_mmw_rA_train, X_mmw_rA_test, Y_train, Y_test = train_test_split(X_mmw_rA, Y, stratify=Y, test_size=0.20,
#                                                                   random_state=3,
#                                                                   shuffle=True)
# del X_mmw_rA

# load model
original_model = tf.keras.models.load_model('../../HPC_User_Study1_Complete_Test/5User_Ultimate_Model/auto_save/train_info/best_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(original_model)  # path to the SavedModel directory
# converter.post_training_quantize=True
tflite_model = converter.convert()

tflite_models_dir = pathlib.Path("indexpen_tflite_models_5User_Ultimate_Model/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir / "indexpen_model.tflite"
tflite_model_file.write_bytes(tflite_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
#
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir / "indexpen_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)

# interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
# interpreter.allocate_tensors()
#
# interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
# interpreter_quant.allocate_tensors()
#
# input_details = interpreter.get_input_details()
# input1_index = interpreter.get_input_details()[0]["index"]
# input2_index = interpreter.get_input_details()[1]["index"]
# output_index = interpreter.get_output_details()[0]["index"]
#
# set_input_time = time.time()
# interpreter.set_tensor(input1_index, np.expand_dims(np.array(X_mmw_rD_test[0]), axis=0).astype(np.float32))
# interpreter.set_tensor(input2_index, np.expand_dims(np.array(X_mmw_rA_test[0]), axis=0).astype(np.float32))
# print('set input time: ', time.time() - set_input_time)
#
# invoke_time = time.time()
# interpreter.invoke()
# print('invoke time: ', time.time() - invoke_time)
#
# get_tensor_time = time.time()
# predictions = interpreter.get_tensor(output_index)
# print('get tensor time: ', time.time() - get_tensor_time)
#
# pred = np.argmax(predictions[0])
# true = np.argmax(Y_test[0])
# #
# print('Test one sample:')
# print(pred)
# print(true)
#
#
# # quantize  all
# # A helper function to evaluate the TF Lite model using "test" dataset.
# # A helper function to evaluate the TF Lite model using "test" dataset.
#
# def evaluate_model(interpreter, dataset='test'):
#     input1_index = interpreter.get_input_details()[0]["index"]
#     input2_index = interpreter.get_input_details()[1]["index"]
#     output_index = interpreter.get_output_details()[0]["index"]
#
#     softmax_outputs = []
#     invoke_durations = []
#     index = 0
#     if dataset == 'test':
#         for X_mmw_rD_sample, X_mmw_rA_sample in zip(X_mmw_rD_test, X_mmw_rA_test):
#             print(index)
#             index += 1
#             interpreter.set_tensor(input1_index, np.expand_dims(np.array(X_mmw_rD_sample), axis=0).astype(np.float32))
#             interpreter.set_tensor(input2_index, np.expand_dims(np.array(X_mmw_rA_sample), axis=0).astype(np.float32))
#
#             ######
#             start_time = time.time()
#             interpreter.invoke()
#             invoke_durations.append(time.time() - start_time)
#             ######
#
#             output = interpreter.tensor(output_index)
#             soft_max_out = np.array(output()[0])
#             softmax_outputs.append(soft_max_out)
#
#     else:
#         for X_mmw_rD_sample, X_mmw_rA_sample in zip(X_mmw_rD_train, X_mmw_rA_train):
#             print(index)
#             index += 1
#             interpreter.set_tensor(input1_index, np.expand_dims(np.array(X_mmw_rD_sample), axis=0).astype(np.float32))
#             interpreter.set_tensor(input2_index, np.expand_dims(np.array(X_mmw_rA_sample), axis=0).astype(np.float32))
#
#             ######
#             start_time = time.time()
#             interpreter.invoke()
#             invoke_durations.append(time.time() - start_time)
#             ######
#
#             output = interpreter.tensor(output_index)
#             soft_max_out = np.array(output()[0])
#             softmax_outputs.append(soft_max_out)
#
#     return softmax_outputs, invoke_durations
#
#
# #
# Y_test_lite_pred, Y_test_lite_test_invoke_durations = evaluate_model(interpreter, dataset='test')
# #
# Y_test_lite_quant_pred, Y_test_lite_quant_test_invoke_durations = evaluate_model(interpreter_quant, dataset='test')
# #
# Y_test_pred = original_model.predict([X_mmw_rD_test, X_mmw_rA_test])
#
#
#
# m = tf.keras.metrics.CategoricalAccuracy()
#
# Y_test_lite_pred_loss = tf.keras.metrics.categorical_crossentropy(Y_test, Y_test_lite_pred).numpy()
# m.update_state(y_true=Y_test, y_pred=Y_test_lite_pred)
# Y_test_lite_pred_acc = m.result().numpy()
# m.reset_state()
#
# Y_test_lite_quant_pred_loss = tf.keras.metrics.categorical_crossentropy(Y_test, Y_test_lite_quant_pred).numpy()
# m.update_state(Y_test, Y_test_lite_quant_pred)
# Y_test_lite_quant_pred_acc = m.result().numpy()
# m.reset_state()
#
# Y_test_pred_loss = tf.keras.metrics.categorical_crossentropy(Y_test, Y_test_pred).numpy()
# m.update_state(Y_test, Y_test_pred)
# Y_test_pred_acc = m.result().numpy()
# m.reset_state()



# print('lite model:')
# print('Accuracy:', Y_test_lite_pred_acc)
# print('Loss:', np.average(Y_test_lite_pred_loss))
# print('Average Invoke Duration:', np.average(Y_test_lite_test_invoke_durations))
#
# print('lite quant model:')
# print('Accuracy:', Y_test_lite_quant_pred_acc)
# print('Loss:', np.average(Y_test_lite_quant_pred_loss))
# print('Average Invoke Duration:', np.average(Y_test_lite_quant_test_invoke_durations))
#
#
# print('original model:')
# print('Accuracy:', Y_test_pred_acc)
# print('Loss:', np.average(Y_test_pred_loss))





