# Authored by Haeyong Kang
from __future__ import print_function

import argparse
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm

from model_nerv import CustomDataSet, Generator
from model_subnet_nerv import SubnetGenerator, SubnetGeneratorMH
from utils import *

import glob
from copy import deepcopy
import wandb

from plots.confusion import conf_matrix, plot_acc_matrix, plot_capacity, plot_psnr, plot_psnr_bpp, plot_psnr_bit

#path = os.path.join(os.path.dirname(__file__), os.pardir)
#sys.path.append(path)

def get_consolidated_masks(per_task_masks, task_id, consolidated_masks=None):

    if task_id == 0:
        consolidated_masks = deepcopy(per_task_masks[task_id])
    else:
        for key in per_task_masks[task_id].keys():
            # Or operation on sparsity
            if consolidated_masks[key] is not None and per_task_masks[task_id][key] is not None:
                consolidated_masks[key] = 1 - ((1 - consolidated_masks[key]) * (1 - per_task_masks[task_id][key]))

    return consolidated_masks


def update_grad(model, consolidated_masks):

    if consolidated_masks is not None and consolidated_masks != {}: 
        # if args.use_continual_masks:
        for key in consolidated_masks.keys():

            if (len(key.split('.')) == 3):
                stem, module, attr = key.split('.')
                module = getattr(getattr(model, stem), module)

            elif (len(key.split('.')) == 4):
                head, layer, module, attr = key.split('.')
                module = getattr(getattr(getattr(model, head), layer), module)

            elif (len(key.split('.')) == 5):
                layers, layer, module1, module2, attr = key.split('.')
                module = getattr(getattr(getattr(getattr(model, layers), layer), module1), module2)

            # Zero-out gradients
            if getattr(module, attr) is not None:
                getattr(module, attr).grad[consolidated_masks[key] == 1] = 0


def main():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--vid',  default=[None], type=int,  nargs='+', help='video id list for training')
    parser.add_argument('--scale', type=int, default=1, help='scale-up facotr for data transformation,  added to suffix!!!!')
    parser.add_argument('--frame_gap', type=int, default=1, help='frame selection gap')
    parser.add_argument('--augment', type=int, default=0, help='augment frames between frames,  added to suffix!!!!')
    parser.add_argument('--dataset', type=str, default='UVG', help='dataset',)
    parser.add_argument('--test_gap', default=1, type=int, help='evaluation gap')

    # NERV architecture parameters
    # embedding parameters
    parser.add_argument('--embed', type=str, default='1.25_80', help='base value/embed length for position encoding')

    # FC + Conv parameters
    parser.add_argument('--stem_dim_num', type=str, default='1024_1', help='hidden dimension and length')
    parser.add_argument('--fc_hw_dim', type=str, default='9_16_128', help='out size (h,w) for mlp')
    parser.add_argument('--expansion', type=float, default=8, help='channel expansion from fc to conv')
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--strides', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list')
    parser.add_argument('--num-blocks', type=int, default=1)

    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    parser.add_argument('--lower-width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument("--single_res", action='store_true', help='single resolution,  added to suffix!!!!')
    parser.add_argument("--conv_type", default='conv', type=str,  help='upscale methods, can add bilinear and deconvolution methods', choices=['conv', 'deconv', 'bilinear'])

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--not_resume_epoch', action='store_true', help='resuming start_epoch from checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--cycles', type=int, default=1, help='epoch cycles for training')
    parser.add_argument('--warmup', type=float, default=0.2, help='warmup epoch ratio compared to the epochs, default=0.2,  added to suffix!!!!')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine', help='learning rate type, default=cosine')
    parser.add_argument('--lr_steps', default=[], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10,  added to suffix!!!!')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for adam. default=0.5,  added to suffix!!!!')
    parser.add_argument('--loss_type', type=str, default='L2', help='loss type, default=L2')
    parser.add_argument('--lw', type=float, default=1.0, help='loss weight,  added to suffix!!!!')
    parser.add_argument('--sigmoid', action='store_true', help='using sigmoid for output prediction')

    # evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--quant_bit', type=int, default=-1, help='bit length for model quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')

    # pruning paramaters
    parser.add_argument('--prune_steps', type=float, nargs='+', default=[0.,], help='prune steps')
    parser.add_argument('--prune_ratio', type=float, default=1.0, help='pruning ratio')

    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:9888', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory,
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")

    parser.add_argument('--n_tasks', type=int, default=7, help='number of tasks')
    parser.add_argument('--subnet', action='store_true', default=False, help='subnet')
    parser.add_argument('--reinit', action='store_true', default=False, help='reinit')
    parser.add_argument('--bias', action='store_true', default=False, help='bias')
    parser.add_argument('--sparsity', '--sparsity', default=0.5, type=float,)

    parser.add_argument('--exp_name', type=str, default='baseline', help='exper name, default=baseline')

    args = parser.parse_args()
    args.warmup = int(args.warmup * args.epochs)

    print(args)
    torch.set_printoptions(precision=4)

    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    if args.prune_ratio < 1 and not args.eval_only:
        prune_str = '_Prune{}_{}'.format(args.prune_ratio, ','.join([str(x) for x in args.prune_steps]))
    else:
        prune_str = ''
    extra_str = '_Strd{}_{}Res{}{}'.format( ','.join([str(x) for x in args.strides]),  'Sin' if args.single_res else f'_lw{args.lw}_multi',
            '_dist' if args.distributed else '', f'_eval' if args.eval_only else '')
    norm_str = '' if args.norm == 'none' else args.norm

    exp_id = f'{args.dataset}/embed{args.embed}_{args.stem_dim_num}_fc_{args.fc_hw_dim}__exp{args.expansion}_reduce{args.reduction}_low{args.lower_width}_blk{args.num_blocks}_cycle{args.cycles}' + \
            f'_gap{args.frame_gap}_e{args.epochs}_warm{args.warmup}_b{args.batchSize}_{args.conv_type}_lr{args.lr}_{args.lr_type}' + \
            f'_{args.loss_type}{norm_str}{extra_str}{prune_str}'
    exp_id += f'_act{args.act}_{args.suffix}'
    args.exp_id = exp_id

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
    	print('Will overwrite the existing output dir!')
    	shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2)
    args.ngpus_per_node = torch.cuda.device_count()

    exp_name = args.exp_name

    if args.subnet:
        args.sparsity = 1 - args.sparsity
        exp_name += '_sparsity' + str(1-args.sparsity)

    if args.bias:
        exp_name += '_bias'

    exp_name += '_fc' + str(args.fc_hw_dim)
    exp_name += '_' + str(args.loss_type)

    if args.reinit:
        exp_name += '_reinit'

    args.exp_name = exp_name

    if 'UVG17' in args.dataset:
        proj_name = 'UVG17'
    else:
        proj_name = args.dataset

    # make exp dir
    os.makedirs('./output/{}'.format(args.exp_name), exist_ok=True)

    #wandb.init(project='NeRV_{}'.format(proj_name),
    #           entity='haeyong', name=exp_name, config=args)

    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)


def train(local_rank, args):

    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    if args.dataset == 'UVG17A':
        data_list = ['./data/bunny', './data/beauty' , './data/bosphorus', './data/bee',
                     './data/jockey', './data/setgo', './data/shake', './data/yacht',
                     './data/city', './data/focus', './data/kids', './data/pan',
                     './data/lips', './data/race', './data/river', './data/sunbath',
                     './data/twilight']

    elif args.dataset == 'UVG17B':
        data_list = [
            './data/bunny',
            './data/city',
            './data/beauty',
            './data/focus',
            './data/bosphorus',
            './data/kids',
            './data/bee',
            './data/pan',
            './data/jockey',
            './data/lips',
            './data/setgo',
            './data/race',
            './data/shake',
            './data/river',
            './data/yacht',
            './data/sunbath',
            './data/twilight'
        ]

    elif args.dataset == 'UVG8':
        data_list = ['./data/bunny', './data/beauty' , './data/bosphorus', './data/bee',
                     './data/jockey', './data/setgo', './data/shake', './data/yacht']

    args.n_tasks = len(data_list)

    PE = PositionalEncoding(args.embed)
    args.embed_length = PE.embed_length

    # define task_masks
    per_task_masks = {}

    if args.subnet:
        model = SubnetGeneratorMH(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num,
                                fc_hw_dim=args.fc_hw_dim, expansion=args.expansion,
                                num_blocks=args.num_blocks, norm=args.norm, act=args.act,
                                bias=args.bias, reduction=args.reduction, conv_type=args.conv_type,
                                stride_list=args.strides,  sin_res=args.single_res,
                                lower_width=args.lower_width, sigmoid=args.sigmoid,
                                sparsity=args.sparsity, n_tasks=args.n_tasks)

    else:
        model = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num,
                          fc_hw_dim=args.fc_hw_dim, expansion=args.expansion,
                          num_blocks=args.num_blocks, norm=args.norm, act=args.act,
                          bias = True, reduction=args.reduction, conv_type=args.conv_type,
                          stride_list=args.strides,  sin_res=args.single_res,
                          lower_width=args.lower_width, sigmoid=args.sigmoid,
                          subnet=args.subnet, sparsity=args.sparsity)

    ##### prune model params and flops #####
    prune_net = args.prune_ratio < 1
    if prune_net:
        param_list = []
        for k,v in model.named_parameters():
            if 'weight' in k:
                if 'stem' in k:
                    stem_ind = int(k.split('.')[1])
                    param_list.append(model.stem[stem_ind])
                elif 'layers' in k[:6] and 'conv' in k:
                    layer_ind = int(k.split('.')[1])
                    param_list.append(model.layers[layer_ind].conv.conv)
        param_to_prune = [(ele, 'weight') for ele in param_list]
        prune_base_ratio = args.prune_ratio ** (1. / len(args.prune_steps))
        args.prune_steps = [int(x * args.epochs) for x in args.prune_steps]
        prune_num = 0
        if args.eval_only:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )

    ##### get model params and flops #####
    total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    if local_rank in [0, None]:
        params = sum([p.data.nelement() for p in model.parameters()]) / 1e6

        print(f'{args}\n {model}\n Model Params: {params}M')
        with open('./output/{}/rank0.txt'.format(args.exp_name), 'a') as f:
            f.write(str(model) + '\n' + f'Params: {params}M\n')
    else:
        writer = None

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()
        args.batchSize = int(args.batchSize / args.ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], \
                                                          output_device=local_rank, find_unused_parameters=False)
    elif args.ngpus_per_node > 1:
        model = torch.nn.DataParallel(model).cuda() #model.cuda() #
    else:
        model = model.cuda()

    # resume from args.weight
    checkpoint = None
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()}
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
            model.load_state_dict(new_ckt)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt)
        else:
            model.load_state_dict(new_ckt)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))

    # resume from model_latest
    checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path) and False:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if prune_net:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )

            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{args.start_epoch}: {sparisity_num / 1e6 / total_params}')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    args.start_epoch = 0
    if checkpoint is not None:
        args.start_epoch = checkpoint['epoch']
        train_best_psnr = checkpoint['train_best_psnr'].to(torch.device(loc))
        train_best_msssim = checkpoint['train_best_msssim'].to(torch.device(loc))
        val_best_psnr = checkpoint['val_best_psnr'].to(torch.device(loc))
        val_best_msssim = checkpoint['val_best_msssim'].to(torch.device(loc))
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.not_resume_epoch:
        args.start_epoch = 0

    # setup dataloader
    img_transforms = transforms.ToTensor()
    DataSet = CustomDataSet

    psnr_matrix = np.zeros((args.n_tasks, args.n_tasks))
    msssim_matrix = np.zeros((args.n_tasks, args.n_tasks))
    taskcla = [(task_id, name.split('/')[-1])for task_id, name in enumerate(data_list)]

    print('*' * 50)
    print(taskcla)

    psnr_matrix = safe_load('./output/{}/psnr_quant{}.npy'.format(args.exp_name, args.quant_bit))
    msssim_matrix = safe_load('./output/{}/msssim_quant{}.npy'.format(args.exp_name, args.quant_bit))

    # plots
    ExNIR = psnr_matrix[-1]
    plot_psnr(ExNIR, dataset=args.dataset)

    com_used_sparsity = safe_load('./output/{}/coused_sparsity.npy'.format(args.exp_name), True)
    global_sparsity = safe_load('./output/{}/global_sparsity.npy'.format(args.exp_name), True)
    reused_sparsity = safe_load('./output/{}/reused_sparsity.npy'.format(args.exp_name), True)


    com_used_init_sparsity = safe_load('./output/{}/coused_sparsity_init.npy'.format(args.exp_name), True)
    global_init_sparsity = safe_load('./output/{}/global_sparsity_init.npy'.format(args.exp_name), True)
    reused_init_sparsity = safe_load('./output/{}/reused_sparsity_init.npy'.format(args.exp_name), True)

    exp_name = args.exp_name + '_psnr'
    plot_acc_matrix(array=psnr_matrix, method=exp_name, dataset=args.dataset)

    exp_name = args.exp_name + '_masssim'
    plot_acc_matrix(array=msssim_matrix, method=exp_name, dataset=args.dataset)

    # plot capacity
    plot_capacity(global_sparsity, reused_sparsity, com_used_sparsity,
                  global_init_sparsity, reused_init_sparsity, com_used_init_sparsity,
                  dataset=args.dataset)

    # plot psnr_bpp
    plot_psnr_bpp(dataset=args.dataset)

    plot_psnr_bit(dataset=args.dataset)

if __name__ == '__main__':
    main()



