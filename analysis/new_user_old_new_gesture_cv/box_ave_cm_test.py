import matplotlib.pyplot as plt
import numpy
import pickle

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

with open('../../HPC_Testing/new_user_old_new_gesture_cross_validation/2_John_20_old_gesture_sample_cv/John_old_gesture_cv_best_cm_acc_hist_dict',
        'rb') as f:
    old_gesture_cm_dict, old_gesture_acc_dict = pickle.load(f)

old_gesture_acc_dataframe = pd.DataFrame.from_dict(old_gesture_acc_dict)
old_gesture_acc_dataframe.boxplot(color='r')
red_patch = mpatches.Patch(color='r', label='John old gesture')

with open('../../HPC_Testing/new_user_old_new_gesture_cross_validation/1_John_20_new_gesture_sample_cv/John_new_gesture_cv_best_cm_acc_hist_dict', 'rb') as f:
    new_gesture_cm_dict, new_gesture_acc_dict = pickle.load(f)

new_gesture_acc_dataframe = pd.DataFrame.from_dict(new_gesture_acc_dict)
new_gesture_acc_dataframe.boxplot(color='b')
blue_patch = mpatches.Patch(color='b', label='John new gesture')



plt.legend(handles=[red_patch, blue_patch])

plt.xlabel('feed in sample ratio from 10 training sample/class in total')
plt.ylabel('testing accuracy from 10 samples/class')
plt.title('new user 20 sample stratify k-fold repeats 20 times Cross-Validation')

plt.show()



#
# normalize = True
#
# print(cm)
# plt.rcParams['xtick.labelsize'] = 20
# plt.rcParams['ytick.labelsize'] = 20
# plt.rcParams['axes.labelsize'] = 12
# plt.rcParams['axes.titlesize'] = 12
# fig, ax = plt.subplots()
# fig.set_size_inches(15, 15)
# im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# ax.figure.colorbar(im, ax=ax)
# # We want to show all ticks...
# ax.set(xticks=np.arange(cm.shape[1]),
#        yticks=np.arange(cm.shape[0]),
#        # ... and label them with the respective list entries
#        xticklabels=classes, yticklabels=classes,
#        title='cm',
#        ylabel='True label',
#        xlabel='Predicted label')
#
# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
#
# # Loop over data dimensions and create text annotations.
# fmt = '.2f' if normalize else 'd'
# fmt_0 = '.0f'
# thresh = cm.max() / 2.
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         ax.text(j, i, format(cm[i, j], fmt if cm[i, j] else fmt_0),
#                 ha="center", va="center",
#                 color="white" if cm[i, j] > thresh else "black")
# fig.tight_layout()