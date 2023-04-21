#!/usr/bin/env bash
export PYTHONPATH=$HOME/Workspaces/PCLIP
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$1"

python3 main_wsn_pmnist.py \
       --config-path configs/task \
       --config-name pmnist.yaml \
       dataset_root="datasets" \
       class_order="class_orders/pmnist.yaml" \
       exp_name="mlp4" \
       seed=1 \
       sparsity=$2


