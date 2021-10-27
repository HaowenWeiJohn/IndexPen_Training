import pickle
import os

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from data_utils.data_general_utils import merge_two_dicts

info = 'Without'
hist_path = '../..' \
           '/HPC_Final_Study1_Complete_Test' \
           '/5User_Ultimate_Model_Without_Clutter_Removal' \
           '/auto_save' \
           '/train_info' \
           '/model_history_log.csv'

hist_data_frame = pd.read_csv(hist_path)



plt.plot(hist_data_frame['epoch'], hist_data_frame['accuracy'])
plt.plot(hist_data_frame['epoch'], hist_data_frame['val_accuracy'])
plt.legend(['train', 'validation'], loc='lower right', fontsize = 13)
plt.xlabel('Epochs', fontsize=13)
plt.ylabel('Accuracy', fontsize=13)
plt.xlim(0,600)
plt.ylim(0,1.05)
plt.title('Accuracy '+ info + ' Clutter Removal')
plt.grid(True, which='both')
plt.savefig('Accuracy '+ info + ' Clutter Removal',dpi=300)
plt.show()


plt.plot(hist_data_frame['epoch'], hist_data_frame['loss'])
plt.plot(hist_data_frame['epoch'], hist_data_frame['val_loss'])
plt.legend(['train', 'validation'], loc='upper left', fontsize = 13)
plt.xlabel('Epochs', fontsize=13)
plt.ylabel('Categorical Cross-Entropy Loss', fontsize=13)
plt.xlim(0,600)
plt.ylim(0, 8)
plt.title('Loss '+ info + ' Clutter Removal')
plt.grid(True, which='both')
plt.savefig('Loss '+ info + ' Clutter Removal',dpi=300)
plt.show()
