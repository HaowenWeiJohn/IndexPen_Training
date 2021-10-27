import pickle
import os

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from data_utils.data_general_utils import merge_two_dicts


instance_path = 'C:/Users/Haowe/PycharmProjects/RealityNavigationRealTimeInference/instance_save/instance.pickle'

with open(instance_path, 'rb') as f:
    data_dict = pickle.load(f)


plt.pcolor(data_dict['rd_cr'])
plt.gca().invert_yaxis()

plt.show()


