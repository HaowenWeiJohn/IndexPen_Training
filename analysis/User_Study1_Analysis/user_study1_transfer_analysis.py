import pickle
import os

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from data_utils.data_general_utils import merge_two_dicts

result_dir = '../../HPC_Final_Study1_Complete_Test'

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)


all_acc_dict = None
legend_patch = []
legend_name = []
for dir in os.listdir(result_dir):
    # C:\Users\Haowe\PycharmProjects\IndexPen_Training\HPC_User_Study1_4User_Complete_Test\Sub3_LeaveOut_test\auto_save\transfer_unfreezeall_info\transfer_learning_best_cm_hist_dict
    if dir.startswith('Sub'):
        subject_index = int(dir[3])
        transfer_learning_best_cm_hist_dict_path = os.path.join(result_dir, dir, 'auto_save',
                                                                'transfer_unfreezeall_info',
                                                                'transfer_learning_best_cm_hist_dict')
        print(transfer_learning_best_cm_hist_dict_path)

        with open(transfer_learning_best_cm_hist_dict_path, 'rb') as f:
            cm_dict, acc_dict = pickle.load(f)

        if all_acc_dict is None:
            all_acc_dict=acc_dict
        else:
            for key in acc_dict:
                test = acc_dict[key]
                all_acc_dict[key] = np.concatenate((all_acc_dict[key], acc_dict[key]))

        acc_dataframe = pd.DataFrame.from_dict(acc_dict)
        acc_dataframe.boxplot(sym='b.', color='C'+str(subject_index))

        legend_patch.append(matplotlib.patches.Patch(color='C'+str(subject_index), label='User'+str(subject_index)))

    plt.legend(handles=legend_patch, loc='lower right', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim((0, 1))


    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.xlabel('feed in sample ratio of 20 samples/class', fontsize=18)
    plt.ylabel('test accuracy for the leave out 180 samples/class', fontsize=18)
    plt.title('Transfer Learning 20 Sample/Class Feed in 10 Fold', fontsize=20)


plt.show()


fig = plt.figure(figsize=(10, 7))
for key in all_acc_dict:
    all_acc_dict[key] = np.mean(all_acc_dict[key])


myList = all_acc_dict.items()
avg_ratio, ave_acc = zip(*myList)
plt.plot(avg_ratio, ave_acc)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim((0, 1))

plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid()
plt.xlabel('feed in sample ratio of 20 samples/class', fontsize=15)
plt.ylabel('test accuracy for the leave out 180 samples/class', fontsize=15)
plt.title('5 User Transfer Learning 20 Sample/Class 10 Fold Average', fontsize=17)


plt.show()
