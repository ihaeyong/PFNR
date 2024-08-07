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

import matplotlib.pyplot as plt

def get_task_sparsity(task_id, per_task_masks, g_sparsity=False):
    curr_task_sparsity = {}
    if g_sparsity:
        curr_task_masks = per_task_masks
    else:
        curr_task_masks = per_task_masks[task_id]

    cum_sparsity = 0
    for key, value in curr_task_masks.items():
        if 'last' in key:
            continue

        if value is not None:
            if 'real' in key or 'imag' in key:

                curr_task_sparsity[key] = []
                for idx in range(len(value)):                
                    curr_task_sparsity[key].append(value[idx].sum() / value[idx].numel())
                    cum_sparsity += value[idx].int().sum()

            else:
                curr_task_sparsity[key] = value.sum() / value.numel()
                cum_sparsity += value.sum()

    return curr_task_sparsity, cum_sparsity


def get_reused_sparsity(task_id, per_task_masks, consolidated_masks):
    curr_reused_sparsity = {}

    if task_id > 0:
        prev_task_masks = per_task_masks[task_id-1]
    else:
        prev_task_masks = per_task_masks[task_id]

    curr_task_masks = per_task_masks[task_id]

    for key, value in curr_task_masks.items():
        if 'last' in key:
            continue

        if 'real' in key or 'imag' in key:
            curr_reused_sparsity[key] = []
            for idx in range(len(prev_task_masks[key])):
                prev_value = prev_task_masks[key][idx]
                if prev_value is not None:
                    value[idx] = prev_value * value[idx]

                if value is not None:
                    curr_reused_sparsity[key].append(value[idx].sum() / value[idx].numel())

        else:
            prev_value = prev_task_masks[key]
            if prev_value is not None:
                value = prev_value * value

            if value is not None:
                curr_reused_sparsity[key] = value.sum() / value.numel()

    return curr_reused_sparsity

def get_coused_sparsity(task_id, per_task_masks, consolidated_masks):

    curr_coused_sparsity = {}
    curr_coused_mask = per_task_masks[task_id]

    if task_id == 0:
        task_id = 1

    for tid in range(task_id):
        prev_task_masks = per_task_masks[tid]

        for key, value in prev_task_masks.items():
            if 'last' in key:
                continue

            if 'real' in key or 'imag' in key:
                curr_coused_sparsity[key] = []
                for idx in range(len(curr_coused_mask[key])):
                    con_value = curr_coused_mask[key][idx]
                    if con_value is not None:
                        value[idx] = con_value * value[idx]
                        curr_coused_mask[key][idx] = value[idx]

                    if value[idx] is not None:
                        curr_coused_sparsity[key].append(value[idx].sum() / value[idx].numel())

            else:
                con_value = curr_coused_mask[key]
                if con_value is not None:
                    value = con_value * value
                    curr_coused_mask[key] = value

                if value is not None:
                    curr_coused_sparsity[key] = value.sum() / value.numel()

    return curr_coused_sparsity



def get_consolidated_masks(per_task_masks, task_id, consolidated_masks=None):

    if task_id == 0:
        consolidated_masks = deepcopy(per_task_masks[task_id])
    else:
        for key in per_task_masks[task_id].keys():
            # Or operation on sparsity
            if consolidated_masks[key] is not None and per_task_masks[task_id][key] is not None:
                if 'real' in key or 'imag' in key:
                    for idx in range(len(consolidated_masks[key])):
                        consolidated_masks[key][idx] = 1 - ((1 - consolidated_masks[key][idx]) * (1 - per_task_masks[task_id][key][idx]))
                else:
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
            if 'real' in key or 'imag' in key:
                    for idx in range(len(consolidated_masks[key])):
                        getattr(module, attr)[idx].grad[consolidated_masks[key][idx] > 0] = 0
            else:
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

    parser.add_argument('--freq', type=int, default=-1, help='freq')
    parser.add_argument('--cat_size', type=int, default=1, help='cat_size')
    parser.add_argument('--lin_type', type=str, default='linear', help='fc type')
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
        if args.freq >= 0:
            if args.cat_size > 0:
                exp_name += '_cat{}'.format(args.cat_size)
            else:
                exp_name += '_sum'
        args.sparsity = 1 - args.sparsity
        exp_name += '_sparsity' + str(1-args.sparsity)
        exp_name += '_{}'.format(args.lin_type)

    if args.bias:
        exp_name += '_bias'

    if args.freq >= 0:
        exp_name += '_freq{}'.format(args.freq)

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
                                  sparsity=args.sparsity, n_tasks=args.n_tasks, device=local_rank,
                                  freq=args.freq, cat_size=args.cat_size, lin_type=args.lin_type)

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

        if False:
            params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
        else:

            params = 0
            for n, p in model.named_parameters():
                if 'w_m' in n:
                    continue
                params+=p.data.nelement()

            params = params / 1e6

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
        model = torch.nn.DataParallel(model).cuda()
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
    print(taskcla)

    train_dataloader_dict = {}
    val_dataloader_dict = {}
    data_size_dict = {}
    train_time_dict = {}
    for task_id, cla in taskcla:
        val_data_dir = data_list[task_id]
        val_dataset = DataSet(val_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.test_gap,  )
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize,  shuffle=False,
                                                     num_workers=args.workers, pin_memory=True,
                                                     sampler=val_sampler, drop_last=False,
                                                     worker_init_fn=worker_init_fn)

        val_dataloader_dict[task_id] = val_dataloader

    print('*' * 50)
    consolidated_masks = None
    global_sparsity = {}
    reused_sparsity = {}
    used_sparsity = {}
    coused_sparsity = {}
    sparsity = args.sparsity

    for task_id, cla in taskcla:

        #if task_id != args.n_tasks -1 :
        #    continue

        print(f'video:{cla}')
        val_dataloader = val_dataloader_dict[task_id]
        checkpoints=torch.load('./output/{}/model_task{}_val_best.pth'.format(args.exp_name, 16)) #args.n_tasks-1))

        #assert task_id == checkpoints['task_id']
        epoch = checkpoints['epoch']
        train_time_dict = checkpoints['train_time']
        state_dict = checkpoints['state_dict']
        train_best_psnr = checkpoints['train_best_psnr']
        train_best_msssim = checkpoints['train_best_msssim']
        val_best_psnr = checkpoints['val_best_psnr']
        val_best_msssim = checkpoints['val_best_msssim']
        per_task_masks = checkpoints['per_task_masks']
        consolidated_masks = checkpoints['consolidated_masks']

        # sparsity
        if False:
            global_sparsity[task_id] = {}
            reused_sparsity[task_id] = {}
            coused_sparsity[task_id] = {}
            used_sparsity[task_id] = {}
            global_sparsity[task_id], used_sparsity[task_id] = get_task_sparsity(task_id, consolidated_masks, g_sparsity=True)
            reused_sparsity[task_id] = get_reused_sparsity(task_id, per_task_masks, consolidated_masks)
            coused_sparsity[task_id] = get_coused_sparsity(task_id, per_task_masks, consolidated_masks)

            print('*' * 50)
            for key, value in global_sparsity[task_id].items():
                if value is not None:
                    re_value = reused_sparsity[task_id][key]
                    cre_value = coused_sparsity[task_id][key]
                    print('task_id{} sparsity : {}, c{}, reused c{}, coused c{}'.format(task_id,
                                                                                        key,
                                                                                        value,
                                                                                        re_value, cre_value))
            print(checkpoints['taskcla'])

            head_params = model.head_layers[0].weight.data.nelement() * (task_id + 1)
            used_sparsity[task_id] = (used_sparsity[task_id] + head_params) / (params * 1e6)

            print('task_id{}, used_sparsity: {}:'.format(task_id, used_sparsity[task_id]))
            print('*' * 50)

            safe_save('./output/{}/global_sparsity'.format(args.exp_name), global_sparsity)
            safe_save('./output/{}/reused_sparsity'.format(args.exp_name), reused_sparsity)
            safe_save('./output/{}/used_sparsity'.format(args.exp_name), used_sparsity)
            safe_save('./output/{}/coused_sparsity'.format(args.exp_name), coused_sparsity)
            
            continue

        model = set_model(model, state_dict, getback=True)

        for task_jd, cla in taskcla:
            val_dataloader = val_dataloader_dict[task_jd]

            if args.dump_images:
                if task_id + 1 < args.n_tasks:
                    continue

            if task_jd == task_id:
                val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args,
                                                per_task_masks, task_id=task_id, task_jd=task_jd, mode='test')
            elif task_jd < task_id:
                #val_psnr, val_msssim = psnr_matrix[task_id-1, task_jd], msssim_matrix[task_id-1, task_jd]
                val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args,
                                                per_task_masks, task_id=task_jd, task_jd=task_jd, mode='test')
            else:
                val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args,
                                                per_task_masks, task_id=task_id, task_jd=task_jd, mode='test')

            psnr_matrix[task_id, task_jd] = val_psnr.item()
            msssim_matrix[task_id, task_jd] = val_msssim.item()

            print('*' * 50)
            print('task_id{}/jd:{}, psnr:{}, msssim:{}'.format(task_id, task_jd, val_psnr.item(), val_msssim.item()))
            print('*' * 50)

            print('PSNR =')
            for i_a in range(task_id+1):
                print('\t',end='')
                for j_a in range(args.n_tasks):
                    print('{:5.2f} '.format(psnr_matrix[i_a, j_a]),end='')
                print()

            print('MSSIM =')
            for i_a in range(task_id+1):
                print('\t',end='')
                for j_a in range(args.n_tasks):
                    print('{:5.2f} '.format(msssim_matrix[i_a, j_a]),end='')
                print()

        del state_dict


    print('*' * 50)
    print(taskcla)

    if not args.dump_images:

        if args.quant_bit != -1:
            safe_save('./output/{}/psnr_quant{}'.format(args.exp_name, args.quant_bit), psnr_matrix)
            safe_save('./output/{}/msssim_quant{}'.format(args.exp_name, args.quant_bit), msssim_matrix)
        else:
            safe_save('./output/{}/psnr'.format(args.exp_name), psnr_matrix)
            safe_save('./output/{}/msssim'.format(args.exp_name), msssim_matrix)


        # PSNR
        print ('Diagonal Final Avg PSNR: {:5.2f}%'.format( np.mean([psnr_matrix[i,i] for i in range(len(taskcla))] )))
        test_avg_psnr = np.mean(psnr_matrix[len(taskcla) - 1])
        print ('Final Avg PSNR: {:5.2f}%'.format( np.mean(psnr_matrix[len(taskcla) - 1])))

        bwt_psnr = np.mean((psnr_matrix[-1]-np.diag(psnr_matrix))[:-1])
        print ('Backward transfer of psnr: {:5.2f}%'.format(bwt_psnr))

        # MSSSIM
        print ('Diagonal Final Avg MSSSIM: {:5.2f}%'.format( np.mean([msssim_matrix[i,i] for i in range(len(taskcla))] )))
        test_avg_msssim = np.mean(msssim_matrix[len(taskcla) - 1])
        print ('Final Avg msssim: {:5.2f}%'.format( np.mean(msssim_matrix[len(taskcla) - 1])))

        bwt_msssim = np.mean((msssim_matrix[-1]-np.diag(msssim_matrix))[:-1])
        print ('Backward transfer of msssim: {:5.2f}%'.format(bwt_msssim))

        total_train_sec = 0
        for key, value in train_time_dict.items():
            total_train_sec += value

        print('[Elapsed traing hours = {:.2f}h]'.format(total_train_sec / 3600))

        log_dict = {
            'test/avg_psnr': test_avg_psnr,
            'test/bwt_psnr': bwt_psnr,
            'test/avg_msssim': test_avg_msssim,
            'text/bwt_msssim': bwt_msssim,
            'test/train_hours': total_train_sec / 3600
        }
        print(log_dict)
        print('-'*50)
        print(taskcla)
        print('-'*50)
        print(args)

@torch.no_grad()
def evaluate(model, val_dataloader, pe, local_rank, args, per_task_masks, task_id, task_jd, mode):
    # Model Quantization
    if args.quant_bit != -1:
        cur_ckt = model.state_dict()
        from dahuffman import HuffmanCodec
        quant_weitht_list = []
        for k,v in cur_ckt.items():
            large_tf = (v.dim() in {2,4} and 'bias' not in k)

            if True: #'real' in k or 'imag' in k:
                quant_v, new_v = quantize_per_tensor(v, args.quant_bit, args.quant_axis if large_tf else -1)
                valid_quant_v = quant_v[v!=0] # only include non-zero weights
                quant_weitht_list.append(valid_quant_v.flatten())
                cur_ckt[k] = new_v
            else:
                pass

        cat_param = torch.cat(quant_weitht_list)
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        # generating HuffmanCoding table
        codec = HuffmanCodec.from_data(input_code_list)
        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        avg_bits = total_bits / len(input_code_list)

        encoding_efficiency = avg_bits / args.quant_bit
        print_str = f'Entropy encoding efficiency for bit {args.quant_bit}: {encoding_efficiency}'
        print(print_str)
        if local_rank in [0, None]:
            with open('./output/{}/eval.txt'.format(args.exp_name), 'a') as f:
                f.write(print_str + '\n')
        model.load_state_dict(cur_ckt)

    psnr_list = []
    msssim_list = []
    if args.dump_images:
        from torchvision.utils import save_image
        visual_dir = f'./output/{args.exp_name}/visualize_{args.quant_bit}/{task_id}/{task_jd}'
        print(f'Saving predictions to {visual_dir}')
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)

    time_list = []
    model.eval()
    for i, (data,  norm_idx) in enumerate(val_dataloader):
        if i > 10 and args.debug:
            break
        embed_input = pe(norm_idx)
        if True:
            task_idx = torch.tensor([(task_id+1) / (args.n_tasks + 1)])
            embed_task = pe(task_idx)
            embed_input = torch.cat([embed_input, embed_task], 1)

        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            embed_input = embed_input.cuda(local_rank, non_blocking=True)
        else:
            data, embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)

        # compute psnr and msssim
        fwd_num = 10 if args.eval_fps else 1
        for _ in range(fwd_num):
            start_time = datetime.now()
            if args.subnet:
                output_list = model(x=embed_input, task_id=task_id, mask=per_task_masks, mode=mode)
            else:
                output_list = model(embed_input)

            torch.cuda.synchronize()
            # torch.cuda.current_stream().synchronize()
            time_list.append((datetime.now() - start_time).total_seconds())

            if args.dump_images:
                stem = model.stem(x=embed_input, task_id=task_id, mask=per_task_masks[task_id], mode=mode).view(embed_input.size(0), model.fc_dim, model.fc_h, model.fc_w)
                l1 = model.layers[0](stem, mask=per_task_masks[task_id], mode=mode)
                l2 = model.layers[1](l1, mask=per_task_masks[task_id], mode=mode)
                l3 = model.layers[2](l2, mask=per_task_masks[task_id], mode=mode)
                l4 = model.layers[3](l3, mask=per_task_masks[task_id], mode=mode)
                l5 = model.layers[4](l4, mask=per_task_masks[task_id], mode=mode)

                l1 = l1.squeeze(0).sum(0) / l1.squeeze(0).shape[0]
                l2 = l2.squeeze(0).sum(0) / l2.squeeze(0).shape[0]
                l3 = l3.squeeze(0).sum(0) / l3.squeeze(0).shape[0]
                l4 = l4.squeeze(0).sum(0) / l4.squeeze(0).shape[0]
                l5 = l5.squeeze(0).sum(0) / l5.squeeze(0).shape[0]

                if False:
                    save_image(l1, f'{visual_dir}/pred_l1.png')
                    save_image(l2, f'{visual_dir}/pred_l2.png')
                    save_image(l3, f'{visual_dir}/pred_l3.png')
                    save_image(l4, f'{visual_dir}/pred_l4.png')
                    save_image(l5, f'{visual_dir}/pred_l5.png')
                else:
                    plt.imshow(l1.cpu())
                    plt.axis('off')
                    plt.savefig(f'{visual_dir}/pred_{i}_l1.pdf', bbox_inches='tight')
                    plt.close()

                    plt.imshow(l2.cpu())
                    plt.axis('off')
                    plt.savefig(f'{visual_dir}/pred_{i}_l2.pdf', bbox_inches='tight')
                    plt.close()

                    plt.imshow(l3.cpu())
                    plt.axis('off')
                    plt.savefig(f'{visual_dir}/pred_{i}_l3.pdf', bbox_inches='tight')
                    plt.close()

                    plt.imshow(l4.cpu())
                    plt.axis('off')
                    plt.savefig(f'{visual_dir}/pred_{i}_l4.pdf', bbox_inches='tight')
                    plt.close()

                    plt.imshow(l5.cpu())
                    plt.axis('off')
                    plt.savefig(f'{visual_dir}/pred_{i}_l5.pdf', bbox_inches='tight')
                    plt.close()

        # dump predictions
        if args.dump_images:
            for batch_ind in range(args.batchSize):
                full_ind = i * args.batchSize + batch_ind
                save_image(output_list[-1][batch_ind], f'{visual_dir}/pred_{full_ind}.png')
                save_image(data[batch_ind], f'{visual_dir}/gt_{full_ind}.png')

        # compute psnr and ms-ssim
        target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
        psnr_list.append(psnr_fn(output_list, target_list))
        msssim_list.append(msssim_fn(output_list, target_list))
        val_psnr = torch.cat(psnr_list, dim=0)              #(batchsize, num_stage)
        val_psnr = torch.mean(val_psnr, dim=0)              #(num_stage)
        val_msssim = torch.cat(msssim_list, dim=0)          #(batchsize, num_stage)
        val_msssim = torch.mean(val_msssim.float(), dim=0)  #(num_stage)
        if i % args.print_freq == 0:
            fps = fwd_num * (i+1) * args.batchSize / sum(time_list)
            print_str = 'Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {} FPS: {}'.format(
                local_rank, i+1, len(val_dataloader),
                RoundTensor(val_psnr, 2, False), RoundTensor(val_msssim, 4, False), round(fps, 2))
            print(print_str)
            if local_rank in [0, None]:
                with open('./output/{}/rank0.txt'.format(args.exp_name), 'a') as f:
                    f.write(print_str + '\n')
    return val_psnr, val_msssim

if __name__ == '__main__':
    main()
