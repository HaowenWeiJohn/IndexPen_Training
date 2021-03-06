import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, axis_font_size=12):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(0,len(classes))))
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['axes.labelsize'] = axis_font_size
    plt.rcParams['axes.titlesize'] = axis_font_size
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    fmt_0 = '.0f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt if cm[i, j] else fmt_0),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm



def plot_dataframe_group_line(data_frame, plot_group=range(0,4),
                          xlabel='Session', ylabel='f1 score',
                              ncol=2,handleheight=2, labelspacing=0.1):

    plt.rcParams['xtick.labelsize'] = 35
    plt.rcParams['ytick.labelsize'] = 35
    plt.rcParams['axes.labelsize'] = 45
    plt.rcParams['axes.titlesize'] = 45


    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111)
    for row in plot_group:
        ax.plot(data_frame.iloc[row],
                linewidth=8,
                markersize=35,
                label=data_frame.index.values[row],
                marker='^', alpha=.5, color='C'+ str(row+1))
    ax.grid()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handleheight=1.2, labelspacing=1, prop={'size': 40})

    # ax.legend(fontsize=35, loc='lower right', ncol=ncol,handleheight=handleheight, labelspacing=labelspacing)

    ax.set_ylim([0, 1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid(linestyle='-', linewidth=2.5)
    # # ax.set_title('None')
    # #     # ax.grid()
    # # plt.ylim(0, 1.1)
    plt.show()
    return ax, fig