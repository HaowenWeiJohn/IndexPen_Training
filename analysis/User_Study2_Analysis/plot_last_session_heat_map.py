import datetime
import glob
import os.path
import pickle
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedShuffleSplit
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from collections import deque
import sys
import seaborn as sns

data_frame_path = 'dataframes/participant_last_session.csv'

data_frame = pd.read_csv(data_frame_path, index_col=0)
data_frame = np.transpose(data_frame)
title = 'Session 5 Gestures F-1 Score'

x_label = data_frame.columns.values.tolist()
y_label = data_frame.index.values.tolist()
normalize = True

data = np.array(data_frame)
cmap=plt.cm.Blues

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.titlesize'] = 25
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
im = ax.imshow(data, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
# We want to show all ticks...
ax.set(xticks=np.arange(data.shape[1]),
       yticks=np.arange(data.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=x_label, yticklabels=y_label,
       title=title,
       ylabel='Participants',
       xlabel='Labels')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = '.2f' if normalize else 'd'
fmt_0 = '.0f'
thresh = data.max() / 2.
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, format(data[i, j], fmt if data[i, j] else fmt_0),
                ha="center", va="center",
                color="white" if data[i, j] > thresh else "black")
fig.tight_layout()

plt.savefig('session_5_gestures_f1_score.png', dpi=300)
plt.show()