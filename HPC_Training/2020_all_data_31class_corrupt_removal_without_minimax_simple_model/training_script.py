import datetime
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_utils.make_model import *
from data_utils.ploting import plot_confusion_matrix

load_data_dir = '2020_31classes_corrupt_frame_removal_(-1000,1500)_(0,2500)'
with open(load_data_dir, 'rb') as f:
    X_mmw_rD, X_mmw_rA, Y, encoder = pickle.load(f)

print(np.min(X_mmw_rD))
print(np.max(X_mmw_rD))
print(np.min(X_mmw_rA))
print(np.max(X_mmw_rA))

rD_min = -1000
rD_max = 1500
rA_min = 0
rA_max = 2500

# X_mmw_rD = (X_mmw_rD - rD_min) / (rD_max - rD_min)
# X_mmw_rA = (X_mmw_rA - rA_min) / (rA_max - rA_min)

X_mmw_rD_train, X_mmw_rD_test, Y_train, Y_test = train_test_split(X_mmw_rD, Y, test_size=0.20, random_state=3,
                                                                  shuffle=True)
del X_mmw_rD

X_mmw_rA_train, X_mmw_rA_test, Y_train, Y_test = train_test_split(X_mmw_rA, Y, test_size=0.20, random_state=3,
                                                                  shuffle=True)
del X_mmw_rA
del Y

model = simple_model(class_num=31)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
csv_logger = CSVLogger("model_history_log.csv", append=True)
mc = ModelCheckpoint(
    # filepath='AutoSave/' + str(datetime.datetime.now()).replace(':', '-').replace(' ',
    filepath=str(datetime.datetime.now()).replace(':', '-').replace(' ',
                                                                    '_') + '.h5',
    monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = model.fit([X_mmw_rD_train, X_mmw_rA_train], Y_train,
                    validation_data=([X_mmw_rD_test, X_mmw_rA_test], Y_test),
                    epochs=2,
                    batch_size=64, callbacks=[es, mc, csv_logger], verbose=1, shuffle=True)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_accuracy.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_loss.png')
plt.show()

best_model_path = glob.glob('./*.h5')[0]
best_model = tf.keras.models.load_model(best_model_path)
Y_pred1 = best_model.predict([X_mmw_rD_test, X_mmw_rA_test])
Y_pred = np.argmax(Y_pred1, axis=1)
Y_test = np.argmax(Y_test, axis=1)
cm = plot_confusion_matrix(y_true=Y_test, y_pred=Y_pred, classes=encoder.categories_[0])
plt.savefig('confusion_matrix.png')
test_acc = accuracy_score(Y_test, Y_pred)
print("best_accuracy_score:", test_acc)

