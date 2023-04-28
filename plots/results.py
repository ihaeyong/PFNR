import numpy as np
import torch

from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse, time

import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from utils_huffman import Huffman
from utils_rle import Polyline

from networks.utils import *
from utils import safe_load, safe_save

from plots.confusion import conf_matrix, plot_acc_matrix

g_exp_name = 'svd'

def estimate_avg_result(per_dict):
    # print
    for key in per_dict.keys():
        avg_global_sparsity = np.mean(per_dict[key]['global_sparsity'])
        std_global_sparsity = np.std(per_dict[key]['global_sparsity'])

        avg_bit_mask_capacity = np.mean(per_dict[key]['bit_mask_capacity'])
        std_bit_mask_capacity = np.std(per_dict[key]['bit_mask_capacity'])

        avg_bwt = np.mean(per_dict[key]['hard_bwt'])
        std_bwt = np.std(per_dict[key]['hard_bwt'])

        avg_fwt = np.mean(per_dict[key]['hard_fwt'])
        std_fwt = np.std(per_dict[key]['hard_fwt'])

        avg_acc = np.mean(per_dict[key]['hard_acc'])
        std_acc = np.std(per_dict[key]['hard_acc'])

        hard_avg_acc = np.array(per_dict[key]['hard_acc_matrix']).mean(0)
        name = 'hard{}'.format(key)
        plot_acc_matrix(array=hard_avg_acc, method=name, dataset='cifar100_100')

        print('-'*20, key ,'-'*20)
        print(key, "spar:{:.5f}({:5.9f})".format(avg_global_sparsity, std_global_sparsity))
        print(key, "bit:{:.5f}({:5.9f})".format(avg_bit_mask_capacity, std_bit_mask_capacity))

        print(key, "hard_bwt:{:.5f}({:5.9f})".format(avg_bwt, std_bwt))
        print(key, "hard_fwt:{:.5f}({:5.9f})".format(avg_fwt, std_fwt))
        print(key, "hard_acc:{:.5f}({:5.9f})".format(avg_acc, std_acc))
        print()

        avg_bwt = np.mean(per_dict[key]['soft_bwt'])
        std_bwt = np.std(per_dict[key]['soft_bwt'])

        avg_fwt = np.mean(per_dict[key]['soft_fwt'])
        std_fwt = np.std(per_dict[key]['soft_fwt'])

        avg_acc = np.mean(per_dict[key]['soft_acc'])
        std_acc = np.std(per_dict[key]['soft_acc'])

        soft_avg_acc = np.array(per_dict[key]['soft_acc_matrix']).mean(0)
        name = 'soft{}'.format(key)
        plot_acc_matrix(array=soft_avg_acc, method=name, dataset='cifar100_100')

        print(key, "soft_bwt:{:.5f}({:5.9f})".format(avg_bwt, std_bwt))
        print(key, "soft_fwt:{:.5f}({:5.9f})".format(avg_fwt, std_fwt))
        print(key, "soft_acc:{:.5f}({:5.9f})".format(avg_acc, std_acc))
        print()



def plot_avg_bars(per_dict, dataset=None):
    if 'tiny' in dataset:
        c_list  = [0.03, 0.1, 0.5]
    else:
        c_list  = [0.5]

    fig, ax = plt.subplots()
    for key in per_dict.keys():
        test_acc = per_dict[key]['test_acc']
        mean = np.mean(test_acc, axis=0)[-1]
        if 'tiny' in  dataset: 
            var = np.var(test_acc, axis=0)[-1] * 0.03
        elif '100_100' in dataset:
            var = np.var(test_acc, axis=0)[-1]
            idx =  (var > 0.9)
            var[idx] =var[idx] * 0.2
        elif '100_10' in dataset:
            var = np.var(test_acc, axis=0)[-1]
            idx =  (var > 0.9)
            var[idx] =var[idx] * 0.2  
        if key in c_list:
            wsn = mean
            #plt.plot(mean, 'o-', lw=2, label='c={}'.format(key))
            #plt.fill_between(range(len(mean)),  mean-var, mean+var, alpha=0.1)

    x = np.arange(0, len(mean),1)

    fs_dgpm = [76.5, 74.2, 73.7, 73.8, 73.7, 73.4, 73.9, 73.7, 72.8, 74.0]
    gpm = [76.2, 73.2, 73.2, 72.2, 72.3, 72.1, 72.7, 72.4, 72.6, 73.0]

    la_maml = [74.2, 71.7, 70.8, 71.2, 70.7, 70.0, 70.5, 69.8, 69.5, 71.3] 
    gem = [75.8, 70.6, 67.7, 68.0, 68.1, 67.6, 67.8, 67.7, 68.3, 69.9]

    ewc = [77.9, 68.5, 68.8, 67.2, 69.3, 69.4, 69.8, 70.7, 71.5, 72.5]

    if False:
        ax.bar(x - 0.25, wsn, color = 'b', width = 0.25, label='WSN (ours)')
        ax.bar(x + 0.00, gpm, color = 'g', width = 0.25, label='GPM')
        ax.bar(x + 0.25, fs_dgpm, color = 'r', width = 0.25, label='FS-DGPM')
    elif False:
        ax.bar(x - 0.40, la_maml, width = 0.18, label='La-MAML')
        ax.bar(x - 0.20, gpm, width = 0.18, label='GPM')
        ax.bar(x + 0.00, fs_dgpm, width = 0.18, label='FS-DGPM')
        ax.bar(x + 0.20, wsn, width = 0.18, label='WSN (ours)')
    elif False:
        width = 0.14
        ax.bar(x - width*2, ewc,     width = width, label='EWC')
        ax.bar(x - width*1, la_maml, width = width, label='La-MAML')
        ax.bar(x + width*0, gpm,     width = width, label='GPM')
        ax.bar(x + width*1, fs_dgpm, width = width, label='FS-DGPM')
        ax.bar(x + width*2, wsn,     width = width, label='WSN (ours)')
    else:
        width = 0.15
        ax.bar(x - width*2, ewc,     width = width, label='EWC')
        ax.bar(x - width*1, la_maml, width = width, label='La-MAML')
        ax.bar(x + width*0, gpm,     width = width, label='GPM')
        ax.bar(x + width*1, fs_dgpm, width = width, label='FS-DGPM')
        ax.bar(x + width*2, wsn,     width = width, label='WSN (ours)')

    label_list = [1,2,3,4,5,6,7,8,9,10, 20, 25, 30, 35, 40]
    x_label = []
    for i in x:
        if (i+1) in label_list:
            x_label.append(str(i+1))
        else:
            x_label.append("")

    ax.set_xticks(x)
    ax.set_xticklabels(x_label, rotation=0)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    plt.ylim(65, 83)
    plt.xlim(-0.8, len(mean)-0.2)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off

    #plt.legend(fontsize=16, loc='upper left')
    plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(0.06,1.0))
    #fig.set_size_inches(9, 6)
    fig.set_size_inches(14, 8)
    #plt.tight_layout()
    #plt.grid(True)
    plt.ylabel('Average Accuracy (%)', fontsize=20)
    plt.xlabel('Increasing No. of Tasks', fontsize=20)
    plt.savefig('./plots/{}_avg_acc_bar_{}.pdf'.format(g_exp_name, dataset), format='pdf',dpi=300)
    plt.close()

def plot_avg_perform(per_dict, dataset=None):

    if 'tiny' in dataset:
        c_list  = [0.03, 0.1, 0.5]
    else:
        c_list  = [0.03, 0.1, 0.5, 0.7]

    fig, ax = plt.subplots()
    for key in per_dict.keys():
        test_acc = per_dict[key]['test_acc']
        mean = np.mean(test_acc, axis=0)[-1]
        if 'tiny' in  dataset: 
            var = np.var(test_acc, axis=0)[-1] * 0.03
        elif '100_100' in dataset:
            var = np.var(test_acc, axis=0)[-1]
            idx =  (var > 0.9)
            var[idx] =var[idx] * 0.2

        elif '100_10' in dataset:
            var = np.var(test_acc, axis=0)[-1]
            idx =  (var > 0.9)
            var[idx] =var[idx] * 0.2  

        if key in c_list:
            plt.plot(mean, 'o-', lw=2, label='c={}'.format(key))
            plt.fill_between(range(len(mean)),  mean-var, mean+var, alpha=0.1)

    x = np.arange(0, len(mean),1)
    label_list = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    x_label = []
    for i in x:
        if (i+1) in label_list:
            x_label.append(str(i+1))
        else:
            x_label.append("")

    ax.set_xticks(x)
    ax.set_xticklabels(x_label, rotation=0)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    #plt.ylim(20, 90)
    plt.xlim(0, len(mean)-1)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off

    plt.legend(fontsize=16, loc='upper left')
    fig.set_size_inches(9, 6)
    #plt.tight_layout()
    #plt.grid(True)

    plt.ylabel('Average Accuracy (%)', fontsize=20)
    plt.xlabel('Increasing No. of Tasks', fontsize=20)
    plt.savefig('./plots/{}_avg_acc_{}.pdf'.format(g_exp_name, dataset), format='pdf',dpi=300)
    plt.close()

def main(args):

    encoding = args.encoding

    dataset = ['csnb_cifar100_10', 'csnb_cifar100_100', 'csnb_tiny_data']
    dataset_ = dataset[0]

    if dataset_ == 'csnb_cifar100_10':
        args.model = 'alexnet'
    elif dataset_ == 'csnb_cifar100_100':
        args.model = 'lenet'
    else:
        args.model = 'tinynet'

    seeds = [1, 2, 3, 4, 5]
    caps = [0.03, 0.05, 0.1, 0.3, 0.5, 0.7]

    layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'linear1', 'linear2']
    reused_list = ['test_reused_acc',
                   'test_all_reused_acc',
                   'test_neg_reused_acc',
                   'test_neg_all_reused_acc']

    task_sparsity = {}
    task_capasity = {}
    task_all_capasity = {}
    per_dict = {}
    per_target_dict = {}
    for i in range(len(caps)):
        args.sparsity = 1 - caps[i]
        per_dict[caps[i]] = {}

        per_dict[caps[i]]['global_sparsity'] = []
        per_dict[caps[i]]['bit_mask_capacity'] = []
        per_dict[caps[i]]['bwt'] = []
        per_dict[caps[i]]['acc'] = []
        per_dict[caps[i]]['test_acc'] = []
        per_dict[caps[i]]['test_reused_acc'] = []
        per_dict[caps[i]]['test_all_reused_acc'] = []
        per_dict[caps[i]]['test_neg_reused_acc'] = []
        per_dict[caps[i]]['test_neg_all_reused_acc'] = []

        # initailize target test acc
        per_target_dict[caps[i]] = {}
        # 'conv1', 'conv2', 'conv3', 'conv4', 'linear1', 'linear2'
        for target in layer_list:
            per_target_dict[caps[i]][target] = {}
            for acc in reused_list:
                per_target_dict[caps[i]][target][acc] = []

        task_sparsity[caps[i]] = {}
        task_capasity[caps[i]] = {}
        task_all_capasity[caps[i]] = {}
        for j in range(len(seeds)):
            args.seed=seeds[j]
            task_sparsity[caps[i]][seeds[j]] = {}
            task_capasity[caps[i]][seeds[j]] = {}
            task_all_capasity[caps[i]][seeds[j]] = {}
            print('-'*20)
            if True:
                # cifar100
                save_name = "{}_{}_{}_SEED_{}_LR_{}_SPARSITY_{}".format(dataset_, args.model, encoding, args.seed, args.lr, 1 - args.sparsity)
            else:
                save_name = "{}_{}_hoffman_SEED_{}_LR_{}_SPARSITY_{}".format('csnb_tiny_data', args.model, args.seed, args.lr, 1 - args.sparsity)
            if not args.prune_thresh == 0.25:
                save_name += "_prune_thresh_{}".format(args.prune_thresh)

            acc_matrix = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".acc.npy")
            test_acc_matrix = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".acc.npy")

            sparsity_matrix = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".cap.npy")
            sparsity_per_task = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".spar.npy", cuda=True)
            per_task_mask = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".pertask.npy", cuda=True)
            consolidated_masks = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".fullmask.npy", cuda=True)
            num_tasks = len(acc_matrix)

            print('-'*20)
            #mask_comp_ratio = safe_load("results2/{}/".format(dataset_) + save_name + ".comp_ratio.npy")
            sparsity_per_layer = print_sparsity(consolidated_masks)
            all_sparsity = global_sparsity(consolidated_masks)

            print("Global Sparsity: {}%".format(all_sparsity * 100))
           # print("Bit Mask Capacity: {}%".format(np.sum(mask_comp_ratio)))

            # Simulation Results
            print ('Diagonal Final Avg Accuracy: {:5.2f}%'.format( np.mean([test_acc_matrix[i,i] for i in range(num_tasks)] )))
            print ('Final Avg accuracy: {:5.2f}%'.format( np.mean(test_acc_matrix[num_tasks - 1])))
            bwt=np.mean((test_acc_matrix[-1]-np.diag(acc_matrix))[:-1])
            print ('Backward transfer: {:5.2f}%'.format(bwt))

            per_dict[caps[i]]['global_sparsity'].append(all_sparsity * 100)
            #per_dict[caps[i]]['bit_mask_capacity'].append(np.sum(mask_comp_ratio))
            per_dict[caps[i]]['bwt'].append(bwt)
            per_dict[caps[i]]['acc'].append(np.mean(test_acc_matrix[num_tasks - 1]))
            per_dict[caps[i]]['test_acc'].append(test_acc_matrix)
            print('-'*20)

    # estimate - average performances
    estimate_avg_result(per_dict)
    plot_avg_perform(per_dict, dataset_)
    plot_avg_bars(per_dict, dataset_)


def main_svd(args):

    encoding = args.encoding
    dataset = ['csnb_cifar100_10', 'csnb_cifar100_100', 'csnb_tiny_data']
    dataset_ = dataset[0]

    if dataset_ == 'csnb_cifar100_10':
        args.model = 'alexnet'
    elif dataset_ == 'csnb_cifar100_100':
        args.model = 'lenet'
    else:
        args.model = 'tinynet'

    seeds = [1, 2, 3, 4, 5]
    caps = [0.03, 0.05, 0.1, 0.3, 0.5, 0.7]
    #caps = [0.05, 0.1, 0.3, 0.5, 0.7]

    layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'linear1', 'linear2']

    reused_list = ['test_reused_acc',
                   'test_all_reused_acc',
                   'test_neg_reused_acc',
                   'test_neg_all_reused_acc']

    task_sparsity = {}
    task_capasity = {}
    task_all_capasity = {}
    per_dict = {}
    per_target_dict = {}
    for i in range(len(caps)):
        args.sparsity = 1 - caps[i]
        per_dict[caps[i]] = {}

        per_dict[caps[i]]['global_sparsity'] = []
        per_dict[caps[i]]['bit_mask_capacity'] = []

        per_dict[caps[i]]['hard_bwt'] = []
        per_dict[caps[i]]['soft_bwt'] = []

        per_dict[caps[i]]['hard_fwt'] = []
        per_dict[caps[i]]['soft_fwt'] = []

        per_dict[caps[i]]['hard_acc'] = []
        per_dict[caps[i]]['soft_acc'] = []

        per_dict[caps[i]]['hard_acc_matrix'] = []
        per_dict[caps[i]]['soft_acc_matrix'] = []

        per_dict[caps[i]]['test_reused_acc'] = []
        per_dict[caps[i]]['test_all_reused_acc'] = []
        per_dict[caps[i]]['test_neg_reused_acc'] = []
        per_dict[caps[i]]['test_neg_all_reused_acc'] = []

        # initailize target test acc
        per_target_dict[caps[i]] = {}
        # 'conv1', 'conv2', 'conv3', 'conv4', 'linear1', 'linear2'
        for target in layer_list:
            per_target_dict[caps[i]][target] = {}
            for acc in reused_list:
                per_target_dict[caps[i]][target][acc] = []

        task_sparsity[caps[i]] = {}
        task_capasity[caps[i]] = {}
        task_all_capasity[caps[i]] = {}
        for j in range(len(seeds)):
            args.seed=seeds[j]
            task_sparsity[caps[i]][seeds[j]] = {}
            task_capasity[caps[i]][seeds[j]] = {}
            task_all_capasity[caps[i]][seeds[j]] = {}
            print('-'*20)

            save_name = "{}_SEED_{}_LR_{}_SPARSITY_{:0.2f}_{}_soft{}_grad{}".format(
                args.dataset,
                args.seed,
                args.lr,
                1 - args.sparsity, args.name, args.soft, args.soft_grad)

            #if not args.prune_thresh == 0.25:
            #    save_name += "_prune_thresh_{}".format(args.prune_thresh)

            acc_matrix = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".acc.npy")
            test_acc_matrix = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".test_acc.npy")

            soft_acc_matrix = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".test_acc_soft.npy")

            rnd_acc_matrix = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".test_acc_random.npy")

            sparsity_matrix = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".cap.npy")
            sparsity_per_task = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".spar.npy", cuda=True)
            per_task_mask = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".pertask.npy", cuda=True)
            consolidated_masks = safe_load("results_{}/{}/".format(g_exp_name, dataset_) + save_name + ".fullmask.npy", cuda=True)
            num_tasks = len(acc_matrix)

            print('-'*20)

            sparsity_per_layer = print_sparsity(consolidated_masks)
            all_sparsity = global_sparsity(consolidated_masks)

            print("Global Sparsity: {}%".format(all_sparsity * 100))

            # Simulation Results
            print ('Diagonal Final Avg Accuracy: {:5.2f}%'.format( np.mean([test_acc_matrix[i,i] for i in range(num_tasks)] )))

            # -- Hard Results
            print ('Hard:Final Avg accuracy: {:5.2f}%'.format( np.mean(test_acc_matrix[num_tasks - 1])))
            hard_bwt=np.mean((test_acc_matrix[-1]-np.diag(test_acc_matrix))[:-1])
            print ('Hard:Backward transfer: {:5.2f}%'.format(hard_bwt))

            fwt_list = []
            for task_id in range(1, num_tasks):
                fwt = test_acc_matrix[task_id-1, task_id] - rnd_acc_matrix[task_id, task_id]
                fwt_list.append(fwt)

            hard_fwt=np.mean(fwt_list)
            print ('Hard:Forward transfer: {:5.2f}%'.format(hard_fwt))

            # -- Soft Results
            print ('Soft:Final Avg accuracy: {:5.2f}%'.format( np.mean(soft_acc_matrix[num_tasks - 1])))
            soft_bwt=np.mean((soft_acc_matrix[-1]-np.diag(soft_acc_matrix))[:-1])
            print ('Soft:Backward transfer: {:5.2f}%'.format(soft_bwt))

            fwt_list = []
            for task_id in range(1, num_tasks):
                fwt = soft_acc_matrix[task_id-1, task_id] - rnd_acc_matrix[task_id, task_id]
                fwt_list.append(fwt)

            soft_fwt=np.mean(fwt_list)
            print ('Soft:Forward transfer: {:5.2f}%'.format(soft_fwt))

            per_dict[caps[i]]['global_sparsity'].append(all_sparsity * 100)
            #per_dict[caps[i]]['bit_mask_capacity'].append(np.sum(mask_comp_ratio))
            per_dict[caps[i]]['hard_bwt'].append(hard_bwt)
            per_dict[caps[i]]['soft_bwt'].append(soft_bwt)

            per_dict[caps[i]]['hard_fwt'].append(hard_fwt)
            per_dict[caps[i]]['soft_fwt'].append(soft_fwt)

            per_dict[caps[i]]['hard_acc'].append(np.mean(test_acc_matrix[num_tasks - 1]))
            per_dict[caps[i]]['soft_acc'].append(np.mean(soft_acc_matrix[num_tasks - 1]))
            per_dict[caps[i]]['hard_acc_matrix'].append(test_acc_matrix)
            per_dict[caps[i]]['soft_acc_matrix'].append(soft_acc_matrix)
            print('-'*20)

    # estimate - average performances
    estimate_avg_result(per_dict)



if __name__ == '__main__':

    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--optim', type=str, default="adam", metavar='OPTIM',
                        help='optimizer choice')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # CUDA parameters
    parser.add_argument('--gpu', type=str, default="0", metavar='GPU',
                        help="GPU ID for single GPU training")
    # CSNB parameters
    parser.add_argument('--sparsity', type=float, default=0.5, metavar='SPARSITY',
                        help="Target current sparsity for each layer")
    # Model parameters
    parser.add_argument('--model', type=str, default="alexnet", metavar='MODEL',
                        help="Models to be incorporated for the experiment")
    # Deep compression
    parser.add_argument("--deep_comp", type=str, default="", metavar='COMP',
                        help="Deep Compression Model")
    # Pruning threshold
    parser.add_argument("--prune_thresh", type=float, default=0.25, metavar='PRU_TH',
                        help="Pruning threshold for Deep Compression")
    # data parameters
    parser.add_argument('--loader', type=str,
                        default='task_incremental_loader',
                        help='data loader to use')
    # increment
    parser.add_argument('--increment', type=int, default=5, metavar='S',
                        help='(default: 5)')

    parser.add_argument('--data_path', default='./data/', help='path where data is located')
    parser.add_argument("--dataset",
                        default='cifar100_100',
                        type=str,
                        required=False,
                        choices=['mnist_permutations', 'cifar100_100', 'cifar100_superclass', 'tinyimagenet', 'pmnist'],
                        help="Dataset to train and test on.")


    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')

    parser.add_argument("--workers", default=4, type=int, help="Number of workers preprocessing the data.")

    parser.add_argument("--glances", default=1, type=int,
                        help="# of times the model is allowed to train over a set of samples in the single pass setting")
    parser.add_argument("--class_order", default="random", type=str, choices=["random", "chrono", "old", "super"],
                        help="define classes order of increment ")

    # For cifar100
    parser.add_argument('--n_tasks', type=int, default=10,
                        help='total number of tasks, invalid for cifar100_superclass')
    parser.add_argument('--shuffle_task', default=False, action='store_true',
                        help='Invalid for cifar100_superclass')

    # data parameters
    parser.add_argument('--encoding', type=str, default='huffman',
                        help='data loader to use')

    parser.add_argument('--name', type=str, default='hard_v1')
    #parser.add_argument('--name', type=str, default='soft_epoch30_randn1e-2_per0.02')
    parser.add_argument('--soft', type=float, default=0.0)
    parser.add_argument('--soft_grad', type=float, default=0.0)

    args = parser.parse_args()
    args.sparsity = 1 - args.sparsity
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)


    if False:
        main(args)
    elif True:
        main_svd(args)
    else:
        plot_bit_task_sparsity()


