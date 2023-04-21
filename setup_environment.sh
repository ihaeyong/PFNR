#!bin/bash

# create enviroment using Miniconda (or Anaconda)
conda create -n clip python=3.8
conda activate clip

# install pytorch
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 \
    --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# install other dependencies
pip install -r requirements.txt

# install CLIP
pip install git+https://github.com/openai/CLIP.git


# torch install
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
