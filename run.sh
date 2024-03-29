#!/usr/bin/env bash
export PYTHONPATH=$HOME/Workspaces/NeRV
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$1"


python train_nerv.py -e 300 \
       --lower-width 96 --num-blocks 1 \
       --dataset bunny --frame_gap 1 \
       --outf bunny_ab --embed 1.25_40 \
       --stem_dim_num 512_1 --reduction 2 \
       --fc_hw_dim 9_16_26 --expansion 1  \
       --single_res --loss Fusion6  --warmup 0.2 \
       --lr_type cosine  --strides 5 2 2 2 2 \
       --conv_type conv \
       -b 1  --lr 0.0005 \
       --norm none --act swish

