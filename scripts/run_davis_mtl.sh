#!/usr/bin/env bash
export PYTHONPATH=$HOME/workspaces/PNR-S
export PYTHONIOENCODING=utf-8
# export CUDA_VISIBLE_DEVICES="$1,$2,$3,$4,$5"
export CUDA_VISIBLE_DEVICES="$1"

if [ $2 == 'train' ]; then
        python train_nerv_mtl.py -e 300 \
                --lower-width 96 --num-blocks 1 \
                --dataset DAVIS50 --frame_gap 1 \
                --outf bunny_ab --embed 1.25_40 \
                --stem_dim_num 512_1 --reduction 2 \
                --fc_hw_dim 9_16_112 --expansion 1  \
                --single_res --loss_type Fusion6  --warmup 0.2 \
                --lr_type cosine  --strides 5 2 2 2 2 \
                --conv_type conv \
                -b 50  --lr 0.0005 \
                --norm none --act swish \
                --cat_size -1 --distributed \
                --exp_name nerv_mtl


elif [ $2 == 'eval' ]; then

        python train_nerv_mtl.py -e 150 \
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
                --freq -1 --cat_size -1 --eval_only \
                --exp_name nerv_mtl
fi


