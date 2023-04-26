#!bin/bash

# create enviroment using Miniconda (or Anaconda)
conda create -n NeRV python=3.8
source activate NeRV

# install pytorch
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 \
    --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# install other dependencies
pip install -r requirements.txt

pip install wandb

# torch install
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
