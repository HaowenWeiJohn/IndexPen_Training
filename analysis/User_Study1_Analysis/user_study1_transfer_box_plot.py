import pickle
import os
from matplotlib import pyplot as plt
import pandas as pd

result_dir = '../../HPC_User_Study1_Complete_Test'

fig = plt.figure(figsize=(10, 7))
# ax = fig.add_axes()

for dir in os.listdir(result_dir):
    # C:\Users\Haowe\PycharmProjects\IndexPen_Training\HPC_User_Study1_4User_Complete_Test\Sub3_LeaveOut_test\auto_save\transfer_unfreezeall_info\transfer_learning_best_cm_hist_dict
    if dir.startswith('Sub2_LeaveOut_test_trouble_shooting'):
        transfer_learning_best_cm_hist_dict_path = os.path.join(result_dir, dir, 'auto_save',
                                                                'transfer_info',
                                                                'transfer_learning_best_cm_hist_dict')
        print(transfer_learning_best_cm_hist_dict_path)

        with open(transfer_learning_best_cm_hist_dict_path, 'rb') as f:
            cm_dict, acc_dict = pickle.load(f)

        acc_dataframe = pd.DataFrame.from_dict(acc_dict)
        acc_dataframe.boxplot()
        # ax.boxplot(acc_dict.values())
        # ax.set_xticklabels(acc_dict.keys())

    plt.xlabel('feed in sample ratio from 10 training sample/class in total')
    plt.ylabel('testing accuracy from 10 samples/class')
    plt.title('new user 20 sample stratify k-fold repeats 20 times Cross-Validation unfreeze all')


# for dir in os.listdir(result_dir):
#     # C:\Users\Haowe\PycharmProjects\IndexPen_Training\HPC_User_Study1_4User_Complete_Test\Sub3_LeaveOut_test\auto_save\transfer_unfreezeall_info\transfer_learning_best_cm_hist_dict
#     if dir.startswith('Sub4_LeaveOut_test_trouble_shooting'):
#         transfer_learning_best_cm_hist_dict_path = os.path.join(result_dir, dir, 'auto_save',
#                                                                 'transfer_unfreezeall_info',
#                                                                 'transfer_learning_best_cm_hist_dict')
#         print(transfer_learning_best_cm_hist_dict_path)
#
#         with open(transfer_learning_best_cm_hist_dict_path, 'rb') as f:
#             cm_dict, acc_dict = pickle.load(f)
#
#
#         for key in acc_dict:
#             new_key = float(key) + 0.02
#             acc_dict[new_key] = acc_dict[key]
#             acc_dict.pop(key)
#
#
#
#
#         acc_dataframe = pd.DataFrame.from_dict(acc_dict)
#         acc_dataframe.boxplot()
#         # ax.boxplot(acc_dict.values())
#         # ax.set_xticklabels(acc_dict.keys())
#
#     plt.xlabel('feed in sample ratio from 10 training sample/class in total')
#     plt.ylabel('testing accuracy from 10 samples/class')
#     plt.title('new user 20 sample stratify k-fold repeats 20 times Cross-Validation unfreeze all')

plt.show()