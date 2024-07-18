#!/usr/bin/env bash
export PYTHONPATH=$HOME/workspaces/PNR-S
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$1"

if [ $2 == 'train' ]; then

    python train_nerv_single.py -e 300 \
           --lower-width 96 --num-blocks 1 \
           --dataset UVG8 --frame_gap 1 \
           --outf bunny_ab --embed 1.25_40 \
           --stem_dim_num 512_1 --reduction 2 \
           --fc_hw_dim 9_16_112 --expansion 1  \
           --single_res --loss_type Fusion6  --warmup 0.2 \
           --lr_type cosine  --strides 5 2 2 2 2 \
           --conv_type conv \
           -b 1  --lr 0.0005 \
           --norm none --act swish \
	         --cat_size -1 --reinit --freq 1 \
           --exp_name nerv_sigle_c0.5


elif [ $2 == 'eval' ]; then

    python train_nerv_eval.py -e 50 \
           --lower-width 96 --num-blocks 1 \
           --dataset DAVIS50 --frame_gap 1 \
           --outf bunny_ab --embed 1.25_40 \
           --stem_dim_num 512_1 --reduction 2 \
           --fc_hw_dim 9_16_112 --expansion 1  \
           --single_res --loss_type Fusion6  --warmup 0.2 \
           --lr_type cosine  --strides 5 2 2 2 2 \
           --conv_type conv \
           -b 1  --lr 0.0005 \
           --norm none --act swish \
           --subnet --sparsity 0.3 \
           --quant_bit $4 \
           --freq -1 --cat_size -1 \
           --exp_name wsn

elif [ $2 == 'plot' ]; then

    python plot_nerv_eval.py -e 50 \
           --lower-width 96 --num-blocks 1 \
           --dataset DAVIS50 --frame_gap 1 \
           --outf bunny_ab --embed 1.25_40 \
           --stem_dim_num 512_1 --reduction 2 \
           --fc_hw_dim 9_16_112 --expansion 1  \
           --single_res --loss_type Fusion6  --warmup 0.2 \
           --lr_type cosine  --strides 5 2 2 2 2 \
           --conv_type conv \
           -b 1 --lr 0.0005 \
           --norm none --act swish \
           --subnet --sparsity 0.3 --reinit \
           --quant_bit $4 \
           --freq 1 --cat_size -1 \
           --exp_name spec

fi
