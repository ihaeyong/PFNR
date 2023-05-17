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
    label_list = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    for i in range(n_tasks):
        if i+1 in label_list:
            task_list.append('S{}'.format(str(i+1)))
        else:
            task_list.append('')

    # confusion matrix
    df_cm = pd.DataFrame(array, index = [i for i in range(n_tasks)],
                         columns = [i for i in range(n_tasks)])

    sn.set(font_scale=1.5)
    s=sn.heatmap(df_cm, annot=True, annot_kws={"size": 15},
                 cbar=False, cbar_kws={"size", 15},
                 #vmin=0, vmax=3000,
                 #square=True,
                 xticklabels=task_list, yticklabels=task_list, fmt='.2f')

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

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(14, 14)

    plt.savefig('./plots/{}/{}_{}.pdf'.format(dataset, dataset, method), format='pdf',dpi=600, bbox_inches='tight')
    plt.close()


def plot_psnr_bit(dataset=None):

    fig, ax = plt.subplots()

    STL_x = [
        0.205, 0.501, 0.752, #1.000
    ]

    STL_y = [
        4.76, 11.08, 10.14, #11.18
    ]

    ExNIR_x = [
        0.082, 0.174, 0.348, 0.697
    ]

    ExNIR_y = [
        25.9, 26.77, 26.80, 26.80
    ]

    ExNIR_reinit_x = [
        0.082, 0.165, 0.292, 0.585
    ]

    ExNIR_reinit_y = [
        25.29, 27.7, 28.89, 28.93
    ]


    MTL_x = [

    ]

    MTL_y = [

    ]

    # STL
    plt.plot(STL_x, STL_y, '-o', lw=1, color='b', label='STL, NeRV')

    # ExNIR
    plt.plot(ExNIR_x, ExNIR_y, marker='o', lw=1, color='c', label='ExNIR (ours)', linestyle='dashed')

    # ExNIR_reinit
    plt.plot(ExNIR_reinit_x, ExNIR_reinit_y, marker='o', lw=1, color='r', label='ExNIR-reinit (ours)', linestyle='solid')

    # MLT
    #plt.plot(MTL_x, MTL_y, '-o', lw=1, color='m', label='MTL')


    ax.set_yticks(np.arange(0, 50, 5))

    plt.legend(fontsize=16, loc='upper left')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(14, 6)

    plt.ylabel('PSNR', fontsize=20)
    plt.xlabel('Bits per pixel (BPP)', fontsize=20)

    plt.savefig('./plots/{}/{}_psnr_bpp_{}.pdf'.format(dataset, dataset, dataset), format='pdf',dpi=600, bbox_inches='tight')
    plt.close()


def plot_psnr_bpp(dataset=None):

    fig, ax = plt.subplots()

    x = [ 4, 8, 16, 32]

    ExNIR_reinit_c10 = [25.29, 27.70, 27.72, 27.72]
    ExNIR_reinit_c30 = [23.82, 26.77, 28.98, 28.98]
    ExNIR_reinit_c50 = [21.51, 28.89, 28.93, 28.93]
    ExNIR_reinit_c70 = [21.64, 27.40, 27.45, 27.45]

    # ExNIR
    #plt.plot(x, ExNIR_reinit_c10, marker='o', lw=1, color='c', label='ExNIR (ours)', linestyle='dashed')

    # ExNIR_reinit
    plt.plot(x, ExNIR_reinit_c10, marker='o', lw=1, color='r', label='ExNIR-reinit, c=10.0%', linestyle='solid')
    plt.plot(x, ExNIR_reinit_c30, marker='o', lw=1, color='c', label='ExNIR-reinit, c=30.0%', linestyle='solid')
    plt.plot(x, ExNIR_reinit_c50, marker='o', lw=1, color='b', label='ExNIR-reinit, c=50.0% ', linestyle='solid')
    plt.plot(x, ExNIR_reinit_c70, marker='o', lw=1, color='m', label='ExNIR-reinit, c=70.0% ', linestyle='solid')


    #x = np.arange(0, len(x),1)
    x_label = []
    for i in x:
        x_label.append(str(i))

    ax.set_xticks(x)
    ax.set_xticklabels(x_label, rotation=0)
    ax.set_yticks(np.arange(20, 40, 5))
    #ax.set_facecolor('white')

    plt.legend(fontsize=16, loc='upper left')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(14, 6)
    
    plt.ylabel('PSNR', fontsize=20)
    plt.xlabel('Bit', fontsize=20)

    plt.savefig('./plots/{}/{}_psnr_bit_{}.pdf'.format(dataset, dataset, dataset), format='pdf',dpi=600, bbox_inches='tight')
    plt.close()


def plot_psnr(ExNIR, dataset=None):

    fig, ax = plt.subplots()

    STL = [
        39.66 , 44.91 , 36.28 , 41.13 , 38.14 , 31.50 , 42.03 ,
        34.76 , 36.59 , 36.85 , 29.23 , 31.79 , 37.27 , 34.15 ,
        31.45 , 38.44 , 43.84
    ]

    ExNIR = [
        31.50, 34.37, 31.00, 32.38, 29.26, 23.08, 31.96, 22.64,
        22.07, 33.48, 18.34, 20.45, 27.21, 24.33, 23.09, 21.23,
        29.13]

    ExNIR_reinit = [
        31.47, 35.42, 32.51, 34.73, 30.70, 24.53, 35.63, 25.49,
        24.50, 35.59, 20.24, 22.49, 30.22, 27.03, 25.14, 24.86,
        32.16
    ]


    MTL = [
        32.39, 34.35, 31.45, 34.03, 30.70, 24.53, 37.13, 27.83,
        23.80, 34.69, 20.77, 22.37, 32.71, 28.00, 25.89, 26.40,
        33.16
    ]

    # STL
    plt.plot(STL, '-o', lw=1, color='b', label='STL, NeRV')

    # ExNIR
    plt.plot(ExNIR, marker='o', lw=1, color='c', label='ExNIR (ours)', linestyle='dashed')

    # ExNIR
    plt.plot(ExNIR_reinit, marker='o', lw=1, color='r', label='ExNIR-reinit (ours)', linestyle='solid')

    # MLT
    plt.plot(MTL, '-o', lw=1, color='m', label='MTL')

    x = np.arange(0, len(ExNIR),1)
    x_label = []
    for i in x:
        x_label.append(str(i+1))

    ax.set_xticks(x)
    ax.set_xticklabels(x_label, rotation=0)
    ax.set_yticks(np.arange(15, 65, 5))

    plt.legend(fontsize=16, loc='upper left')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(14, 6)

    plt.ylabel('PSNR', fontsize=20)
    plt.xlabel('Sessions', fontsize=20)

    plt.savefig('./plots/{}/{}_psnr_{}.pdf'.format(dataset, dataset, dataset), format='pdf',dpi=600, bbox_inches='tight')
    plt.close()

def plot_capacity(global_s, reused_s, comm_s,
                  global_s_init, reused_s_init, comm_s_init,
                  dataset=None):

    color = {
        '0.102': '#1f77b4',
        #'0.3': 'c',
        '0.5': '#ff7f0e',
        #'0.7': 'y',
        '0.9': '#2ca02c'
    }

    l_type = {
        'stem.mlp_fc1.weight': 'o',
        #'stem.mlp_fc2.weight': '>',
        'layers.1.conv.conv.weight': 'v',
        #'layers.2.conv.conv.weight': '<',
        #'layers.3.conv.conv.weight': '*',
        'layers.4.conv.conv.weight': 's',
    }

    fig, ax = plt.subplots()

    for key in global_s[0].keys():

        if key not in l_type:
            continue

        g_list, g_init_list = [], []
        r_list, r_init_list = [], []
        c_list, c_init_list = [], []
        for task_id in global_s.keys():

            global_v = global_s[task_id][key].item() * 100.0
            reused_v = reused_s[task_id][key].item() * 100.0
            common_v = comm_s[task_id][key].item() * 100.0

            g_list.append(global_v)
            r_list.append(reused_v)
            c_list.append(common_v)


            global_v = global_s_init[task_id][key].item() * 100.0
            reused_v = reused_s_init[task_id][key].item() * 100.0
            common_v = comm_s_init[task_id][key].item() * 100.0

            g_init_list.append(global_v)
            r_init_list.append(reused_v)
            c_init_list.append(common_v)


            print(f'task_idx{task_id}, {key}: g{global_v}, r{reused_v}, c{common_v}')

        # reinit
        plt.plot(g_list, l_type[key], lw=1, color='r', label=key + '_reinit', linestyle='solid')
        plt.plot(r_list, l_type[key], lw=1, color='g', linestyle='solid')

        # init
        plt.plot(g_init_list, marker=l_type[key], lw=1, color='c', linestyle='dashed')
        plt.plot(r_init_list, marker=l_type[key], lw=1, color='g', linestyle='dashed')

    x = np.arange(0, len(g_list),1)
    x_label = []
    for i in x:
        x_label.append(str(i+1))

    ax.set_xticks(x)
    ax.set_xticklabels(x_label, rotation=0)
    ax.set_yticks(np.arange(10, 100, 5))

    plt.legend(fontsize=16, loc='upper left')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(14, 6)

    plt.ylabel('Capacity(%)', fontsize=20)
    plt.xlabel('Sessions', fontsize=20)

    plt.savefig('./plots/{}/{}_reinit_capacity_{}.pdf'.format(dataset, dataset, dataset), format='pdf',dpi=600, bbox_inches='tight')
    plt.close()



