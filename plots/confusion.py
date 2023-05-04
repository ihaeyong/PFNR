import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sn
import os

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data.astype(np.int), **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def conf_matrix(data=None, method='Barlow_Twin', dataset=None):

    if data is None :
        n_tasks=7
        data = np.random.randint(2, 100, size=(n_tasks, n_tasks))
    else:
        n_tasks = data.shape[0]

    #fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
    fig, ax = plt.subplots()
    y = ["task {}".format(i) for i in range(1, n_tasks+1)]
    x = ["task {}".format(i) for i in range(1, n_tasks+1)]
    #x = ["task {}".format(i) for i in list("ABCDEFG")]

    im, _ = heatmap(data, y, x, ax=ax, vmin=0,
                    cmap="magma_r", cbarlabel="{}".format(dataset))

    annotate_heatmap(im, valfmt="{x:d}", size=7, threshold=20,
                     textcolors=("red", "white"))

    plt.savefig('./plots/{}.pdf'.format(method), dpi=600, format='pdf', bbox_inches='tight')

    plt.close()

def plot_acc_matrix(array=None, method='Barlow_Twin', dataset=None):

    n_tasks = array.shape[0]

    # set task_list
    task_list = []
    #label_list = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    for i in range(n_tasks):
        #if i+1 in label_list:
        task_list.append('T{}'.format(str(i+1)))
        #else:
        #    task_list.append('')

    # confusion matrix
    df_cm = pd.DataFrame(array, index = [i for i in range(n_tasks)],
                         columns = [i for i in range(n_tasks)])

    sn.set(font_scale=1.3)
    s=sn.heatmap(df_cm, annot=True, annot_kws={"size": 14},
                 cbar=False, cbar_kws={"size", 14},
                 #vmin=40, vmax=70,
                 xticklabels=task_list, yticklabels=task_list, fmt='.1f')

    s.set_xticklabels(s.get_xticklabels(), rotation = 0, fontsize = 16)
    s.set_yticklabels(s.get_yticklabels(), rotation = 0, fontsize = 16)

    if False:
        # use matplotlib.colorbar.Colorbar object
        cbar = s.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=14)

    path = "./plots/{}".format(dataset)
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!: {}".format(path))

    plt.savefig('./plots/{}/{}_{}.pdf'.format(dataset, dataset, method), format='pdf',dpi=600)
    plt.close()

