# Authored by Haeyong Kang
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from model.subnet import SubnetConv2d, SubnetLinear, SubnetConvTranspose2d

from neuralop.spectral_convolution import FactorizedSpectralConv as SpectralConv
#from neuralop.spectral_convolution import FactorizedSpectralConvV1 as SpectralConv
from neuralop.spectral_convolution import FactorizedSpectralConv2d as SpectralConv2D
from neuralop.spectral_linear import FactorizedSpectralLinear as SpectralLinear
from neuralop.fno_block import FNOBlocks

from neuralop.resample import resample
from neuralop.skip_connections import skip_connection


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, vid_list=[None], frame_gap=1,  visualize=False):
        self.main_dir = main_dir
        self.transform = transform
        frame_idx, self.frame_path = [], []
        accum_img_num = []
        all_imgs = os.listdir(main_dir)
        all_imgs.sort()

        num_frame = 0
        for img_id in all_imgs:
            self.frame_path.append(img_id)
            frame_idx.append(num_frame)
            num_frame += 1

        accum_img_num.append(num_frame)
        self.frame_idx = [float(x) / len(frame_idx) for x in frame_idx]
        self.accum_img_num = np.asfarray(accum_img_num)
        if None not in vid_list:
            self.frame_idx = [self.frame_idx[i] for i in vid_list]
        self.frame_gap = frame_gap

    def __len__(self):
        return len(self.frame_idx) // self.frame_gap

    def __getitem__(self, idx):
        valid_idx = idx * self.frame_gap
        img_id = self.frame_path[valid_idx]
        img_name = os.path.join(self.main_dir, img_id)
        image = Image.open(img_name).convert("RGB")
        tensor_image = self.transform(image)
        if tensor_image.size(1) > tensor_image.size(2):
            tensor_image = tensor_image.permute(0,2,1)
        frame_idx = torch.tensor(self.frame_idx[valid_idx])

        return tensor_image, frame_idx

class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)


def ActivationLayer(act_type):

    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = torch.sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):

    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class CustomConv(nn.Module):
    def __init__(self, **kargs):
        super(CustomConv, self).__init__()

        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.name = kargs['name']
        self.sparsity = kargs['sparsity']
        self.conv_type = kargs['conv_type']
        self.cat_size = kargs['cat_size']

        self.device = kargs['device']
        self.scale = 0.1
        self.var = 0.5
        self.idx_th = 0
        
        self.temp = 1
        self.softmax = nn.Softmax(dim=1)

        if self.conv_type == 'conv':
            self.conv = SubnetConv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'], sparsity=self.sparsity)
            self.up_scale = nn.PixelShuffle(stride)

        elif self.conv_type == 'deconv':
            self.conv = SubnetConvTranspose2d(ngf, new_ngf,
                                              kernel_size=2,
                                              stride=stride,
                                              padding=0,
                                              sparsity=self.sparsity)
            self.up_scale = nn.Identity()

        elif self.conv_type == 'bilinear':
            if not self.subnet:
                self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
                self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])
            else:
                self.conv = None
                self.up_scale = None

        elif self.conv_type == 'convfreq' :
            if self.name == 'layers.0.conv':
                n_modes = (9,16)
            elif self.name == 'layers.1.conv':
                n_modes = (45,80)
            elif self.name == 'layers.2.conv':
                n_modes = (90,160)
            elif self.name == 'layers.3.conv':
                n_modes = (180,320)

            self.up_scale = nn.Identity()

            factorization='subnet'
            rank=1.0
            self.n_layers = 1
            output_scaling_factor = (stride, stride)
            incremental_n_modes = None
            fft_norm = 'forward'
            fixed_rank_modes=False
            implementation = 'factorized'
            separable=False
            decomposition_kwargs = dict()
            joint_factorization = False
            self.conv = SpectralConv(
                ngf,  new_ngf // self.n_layers,
                n_modes,
                output_scaling_factor=output_scaling_factor,
                incremental_n_modes=incremental_n_modes,
                rank=rank,
                fft_norm=fft_norm,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                joint_factorization=joint_factorization,
                n_layers=self.n_layers, bias=False, sparsity=self.sparsity,
                device=self.device, scale=self.scale, 
                idx_th=self.idx_th, var=self.var)

        elif self.conv_type == 'convfreq_sum' :
            if self.name == 'layers.0.conv':
                n_modes = (9,16)
            elif self.name == 'layers.1.conv':
                n_modes = (45,80)
            elif self.name == 'layers.2.conv':
                n_modes = (90,160)
            elif self.name == 'layers.3.conv':
                n_modes = (180,320)

            #self.conv = SubnetConv2d(ngf,
            #                         new_ngf * stride * stride, 3, 1, 1,
            #                         bias=kargs['bias'], sparsity=self.sparsity)

            self.up_scale = nn.PixelShuffle(stride)

            factorization='subnet'
            rank=1.0
            self.n_layers = 1
            output_scaling_factor = (stride, stride)
            incremental_n_modes = None
            fft_norm = 'forward'
            fixed_rank_modes=False
            implementation = 'factorized'
            separable=False
            decomposition_kwargs = dict()
            joint_factorization = False

            # freq-sparsity
            self.fft_scale = 1.0
            fft_sparsity = 1 - (1-self.sparsity) * self.fft_scale

            self.conv_freq = SpectralConv(
                ngf,  new_ngf // self.n_layers,
                n_modes,
                output_scaling_factor=output_scaling_factor,
                incremental_n_modes=incremental_n_modes,
                rank=rank,
                fft_norm=fft_norm,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                joint_factorization=joint_factorization,
                n_layers=self.n_layers, bias=False,
                sparsity=fft_sparsity,
                device=self.device, scale=self.scale,
                idx_th=self.idx_th, var=self.var)

        elif self.conv_type == 'convfreq_cat' :
            if self.name == 'layers.0.conv':
                n_modes = (9,16)
            elif self.name == 'layers.1.conv':
                n_modes = (45,80)
            elif self.name == 'layers.2.conv':
                n_modes = (90,160)
            elif self.name == 'layers.3.conv':
                n_modes = (180,320)

            freq_new_ngf = self.cat_size
            conv_new_ngf = new_ngf - freq_new_ngf

            self.conv = SubnetConv2d(ngf,
                                     conv_new_ngf * stride * stride, 3, 1, 1,
                                     bias=kargs['bias'], sparsity=self.sparsity)

            self.up_scale = nn.PixelShuffle(stride)

            factorization='subnet'
            rank=1.0
            self.n_layers = 1
            output_scaling_factor = (stride, stride)
            incremental_n_modes = None
            fft_norm = 'forward'
            fixed_rank_modes=False
            implementation = 'factorized'
            separable=False
            decomposition_kwargs = dict()
            joint_factorization = False
            self.conv_freq = SpectralConv(
                ngf,  freq_new_ngf // self.n_layers,
                n_modes,
                output_scaling_factor=output_scaling_factor,
                incremental_n_modes=incremental_n_modes,
                rank=rank,
                fft_norm=fft_norm,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                joint_factorization=joint_factorization,
                n_layers=self.n_layers, bias=False, sparsity=self.sparsity,
                device=self.device, scale=self.scale, 
                idx_th=self.idx_th, var=self.var)

            # cat, 0.9
            # conv : 2500, 112, 3, 3 = 2,520,000
            # conv_freq : 43,008 * 4 =   171,032 --> 2,692,032

            # sum
            # conv: 2,825,200
            # conv_freq: 1,605,632 -->  

    def forward(self, x, task_id=None, mask=None, mode="train"):

        if self.conv_type == 'convfreq':
            if mask is not None:
                name = self.name + '.conv'
                weight_mask_real = mask[name + '.weight_real']
                weight_mask_imag = mask[name + '.weight_imag']
            else:
                weight_mask_real = None
                weight_mask_imag = None

            if self.n_layers == 2:
                o1 = self.conv(x=x,indices=0, output_shape=None,
                               weight_mask_real=weight_mask_real,
                               weight_mask_imag=weight_mask_imag,
                               mode=mode)

                o2 = self.conv(x=x, indices=1, output_shape=None,
                               weight_mask_real=weight_mask_real,
                               weight_mask_imag=weight_mask_imag,
                               mode=mode)
                out = torch.cat((o1, o2), axis=1)

            elif self.n_layers == 1:
                out = self.conv(x=x,indices=0, output_shape=None,
                                weight_mask_real=weight_mask_real,
                                weight_mask_imag=weight_mask_imag,
                                mode=mode)

        elif self.conv_type == 'convfreq_sum_backup':

            if mask is not None:
                name = self.name + '.conv'
                weight_mask = mask[name + '.weight']
                bias_mask = mask[name + '.bias']
            else:
                weight_mask = None
                bias_mask = None

            out = self.conv(x, weight_mask=weight_mask, bias_mask=bias_mask, mode=mode)
            out = self.up_scale(out)

            if mask is not None:
                name = self.name + '.conv_freq'
                weight_mask_real = mask[name + '.weight_real']
                weight_mask_imag = mask[name + '.weight_imag']
            else:
                weight_mask_real = None
                weight_mask_imag = None

            out += self.conv_freq(x=x,indices=0, output_shape=None,
                                  weight_mask_real=weight_mask_real,
                                  weight_mask_imag=weight_mask_imag,
                                  mode=mode)

            return out


        elif self.conv_type == 'convfreq_sum':

            #if mask is not None:
            #    name = self.name + '.conv'
            #    weight_mask = mask[name + '.weight']
            #    bias_mask = mask[name + '.bias']
            #else:
            #    weight_mask = None
            #    bias_mask = None

            # out = self.conv(x, weight_mask=weight_mask, bias_mask=bias_mask, mode=mode)
            # out = self.up_scale(out)

            if mask is not None:
                name = self.name + '.conv_freq'
                weight_mask_real = mask[name + '.weight_real']
                weight_mask_imag = mask[name + '.weight_imag']
            else:
                weight_mask_real = None
                weight_mask_imag = None

            out = self.conv_freq(x=x,indices=0, output_shape=None,
                                 weight_mask_real=weight_mask_real,
                                 weight_mask_imag=weight_mask_imag,
                                 mode=mode)

            return out


        elif self.conv_type == 'convfreq_cat':

            if mask is not None:
                name = self.name + '.conv'
                weight_mask = mask[name + '.weight']
                bias_mask = mask[name + '.bias']
            else:
                weight_mask = None
                bias_mask = None

            out = self.conv(x, weight_mask=weight_mask, bias_mask=bias_mask, mode=mode)
            out1 = self.up_scale(out)

            if mask is not None:
                name = self.name + '.conv_freq'
                weight_mask_real = mask[name + '.weight_real']
                weight_mask_imag = mask[name + '.weight_imag']
            else:
                weight_mask_real = None
                weight_mask_imag = None

            out2 = self.conv_freq(x=x,indices=0, output_shape=None,
                                 weight_mask_real=weight_mask_real,
                                 weight_mask_imag=weight_mask_imag,
                                 mode=mode)

            out = torch.cat((out1, out2), dim=1)

            return out


        elif self.conv_type == 'deconv':
            if mask is not None:
                name = self.name + '.conv'
                weight_mask = mask[name + '.weight']
                bias_mask = mask[name + '.bias']
            else:
                weight_mask = None
                bias_mask = None

            # x [1, 122, 45, 80]
            out = self.conv(x, weight_mask=weight_mask, bias_mask=bias_mask, mode=mode)
            # out: [1, 96, 90, 160]
        else:
            if mask is not None:
                name = self.name + '.conv'
                weight_mask = mask[name + '.weight']
                bias_mask = mask[name + '.bias']
            else:
                weight_mask = None
                bias_mask = None

            out = self.conv(x, weight_mask=weight_mask, bias_mask=bias_mask, mode=mode)


        # i=0, [1, 2800, 9, 16]
        # i=1, [1, 384, 45, 80]
        # i=2, [1, 384, 90, 160]
        # i=3, [1, 384, 180, 320]
        out = self.up_scale(out)
        # i=0, [1, 112, 45, 80]
        # i=1,
        # i=2,
        # i=3,
        # i=4,

        return out

class SubnetMLP(nn.Module):
    def __init__(self, **kargs):
        super(SubnetMLP, self).__init__()

        self.name = kargs['name']
        self.act_fn = ActivationLayer('relu')
        self.dim_list = kargs['dim_list']
        self.bias = kargs['bias']
        self.sparsity = kargs['sparsity']
        self.lin_type = kargs['lin_type']


        self.device = kargs['device']
        self.scale = 0.1
        self.var = 0.5
        self.idx_th = 0

        assert len(self.dim_list)-1 == 2


        if self.lin_type == 'linear':

            self.mlp_fc1 = SubnetLinear(self.dim_list[0], self.dim_list[1],
                                        bias=self.bias, sparsity=self.sparsity)

            self.mlp_fc2 = SubnetLinear(self.dim_list[1], self.dim_list[2],
                                        bias=self.bias, sparsity=self.sparsity)

        elif self.lin_type == 'freq4f':

            n_modes = (2,)
            factorization='subnet'
            rank=1.0
            self.n_layers = 1
            output_scaling_factor = (1,1)
            incremental_n_modes = None
            fft_norm = 'forward'
            fixed_rank_modes=False
            implementation = 'factorized'
            separable=False
            decomposition_kwargs = dict()
            joint_factorization = False


            self.mlp_fc1 = SpectralConv(
                self.dim_list[0],  self.dim_list[1] // self.n_layers,
                n_modes,
                output_scaling_factor=output_scaling_factor,
                incremental_n_modes=incremental_n_modes,
                rank=rank,
                fft_norm=fft_norm,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                joint_factorization=joint_factorization,
                n_layers=self.n_layers, bias=False, sparsity=self.sparsity,
                device=self.device, scale=self.scale,
                idx_th=self.idx_th, var=self.var)


            self.mlp_fc2 = SubnetLinear(self.dim_list[1], self.dim_list[2],
                                        bias=self.bias, sparsity=self.sparsity)


        elif self.lin_type == 'freq4s':

            self.mlp_fc1 = SubnetLinear(self.dim_list[0], self.dim_list[1],
                                        bias=self.bias, sparsity=self.sparsity)

            n_modes = (2,)
            factorization='subnet'
            rank=1.0
            self.n_layers = 1
            output_scaling_factor = (1,)
            incremental_n_modes = None
            fft_norm = 'forward'
            fixed_rank_modes=False
            implementation = 'factorized'
            separable=False
            decomposition_kwargs = dict()
            joint_factorization = False

            self.mlp_fc2 = SpectralConv(
                self.dim_list[1],  self.dim_list[2] // self.n_layers,
                n_modes,
                output_scaling_factor=output_scaling_factor,
                incremental_n_modes=incremental_n_modes,
                rank=rank,
                fft_norm=fft_norm,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                joint_factorization=joint_factorization,
                n_layers=self.n_layers, bias=False, sparsity=self.sparsity,
                device=self.device, scale=self.scale,
                idx_th=self.idx_th, var=self.var)


        elif self.lin_type == 'freq4all':

            n_modes = (2, 2)
            factorization='subnet'
            rank=1.0
            self.n_layers = 1
            output_scaling_factor = (1, 1)
            incremental_n_modes = None
            fft_norm = 'forward'
            fixed_rank_modes=False
            implementation = 'factorized'
            separable=False
            decomposition_kwargs = dict()
            joint_factorization = False


            self.mlp_fc1 = SpectralConv(
                self.dim_list[0],  self.dim_list[1] // self.n_layers,
                n_modes,
                output_scaling_factor=output_scaling_factor,
                incremental_n_modes=incremental_n_modes,
                rank=rank,
                fft_norm=fft_norm,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                joint_factorization=joint_factorization,
                n_layers=self.n_layers, bias=False, sparsity=self.sparsity,
                device=self.device, scale=self.scale,
                idx_th=self.idx_th, var=self.var)


            self.mlp_fc2 = SpectralConv(
                self.dim_list[1],  self.dim_list[2] // self.n_layers,
                n_modes,
                output_scaling_factor=output_scaling_factor,
                incremental_n_modes=incremental_n_modes,
                rank=rank,
                fft_norm=fft_norm,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                joint_factorization=joint_factorization,
                n_layers=self.n_layers, bias=False, sparsity=self.sparsity,
                device=self.device, scale=self.scale,
                idx_th=self.idx_th, var=self.var)


    def forward(self, x, task_id=None, mask=None, mode="train"):

        if self.lin_type == 'linear':

            if mask is not None:
                name = self.name + ".mlp_fc1"
                weight_mask = mask[name + '.weight']
                bias_mask = mask[name + '.bias']
            else:
                weight_mask = None
                bias_mask = None

            x = self.mlp_fc1(x, weight_mask=weight_mask, bias_mask=bias_mask, mode=mode)
            x = self.act_fn(x)

            if mask is not None:
                name = self.name + ".mlp_fc2"
                weight_mask = mask[name + '.weight']
                bias_mask = mask[name + '.bias']
            else:
                weight_mask = None
                bias_mask = None

            x = self.mlp_fc2(x, weight_mask=weight_mask, bias_mask=bias_mask, mode=mode)
            x = self.act_fn(x)

        elif self.lin_type == 'freq4f':

            if mask is not None:
                name = self.name + ".mlp_fc1"
                weight_mask_real = mask[name + '.weight_real']
                weight_mask_imag = mask[name + '.weight_imag']
            else:
                weight_mask_real = None
                weight_mask_imag = None

            x = self.mlp_fc1(x=x[:,:,None],indices=0, output_shape=None,
                             weight_mask_real=weight_mask_real,
                             weight_mask_imag=weight_mask_imag,
                             mode=mode)

            x = self.act_fn(x).squeeze(2)

            if mask is not None:
                name = self.name + ".mlp_fc2"
                weight_mask = mask[name + '.weight']
                bias_mask = mask[name + '.bias']
            else:
                weight_mask = None
                bias_mask = None

            x = self.mlp_fc2(x, weight_mask=weight_mask, bias_mask=bias_mask, mode=mode)
            x = self.act_fn(x)


        elif self.lin_type == 'freq4s':

            if mask is not None:
                name = self.name + ".mlp_fc1"
                weight_mask = mask[name + '.weight']
                bias_mask = mask[name + '.bias']
            else:
                weight_mask = None
                bias_mask = None

            x = self.mlp_fc1(x, weight_mask=weight_mask, bias_mask=bias_mask, mode=mode)
            x = self.act_fn(x)

            if mask is not None:
                name = self.name + ".mlp_fc2"
                weight_mask_real = mask[name + '.weight_real']
                weight_mask_imag = mask[name + '.weight_imag']
            else:
                weight_mask_real = None
                weight_mask_imag = None

            x = self.mlp_fc2(x=x[:,:,None],indices=0, output_shape=None,
                             weight_mask_real=weight_mask_real,
                             weight_mask_imag=weight_mask_imag,
                             mode=mode)

            x = self.act_fn(x).squeeze(2)

        elif self.lin_type == 'freq4all':

            if mask is not None:
                name = self.name + ".mlp_fc1"
                weight_mask_real = mask[name + '.weight_real']
                weight_mask_imag = mask[name + '.weight_imag']
            else:
                weight_mask_real = None
                weight_mask_imag = None

            x = self.mlp_fc1(x=x[:,:,None],indices=0, output_shape=None,
                             weight_mask_real=weight_mask_real,
                             weight_mask_imag=weight_mask_imag,
                             mode=mode)
            x = self.act_fn(x)

            if mask is not None:
                name = self.name + ".mlp_fc2"
                weight_mask_real = mask[name + '.weight_real']
                weight_mask_imag = mask[name + '.weight_imag']
            else:
                weight_mask_real = None
                weight_mask_imag = None

            x = self.mlp_fc2(x=x,indices=0, output_shape=None,
                             weight_mask_real=weight_mask_real,
                             weight_mask_imag=weight_mask_imag,
                             mode=mode)

            x = self.act_fn(x).squeeze(2)

        return x

class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.name = kargs['name']
        self.conv = CustomConv(ngf=kargs['ngf'],
                               new_ngf=kargs['new_ngf'],
                               stride=kargs['stride'],
                               bias=kargs['bias'],
                               conv_type=kargs['conv_type'],
                               sparsity=kargs['sparsity'],
                               device=kargs['device'],
                               cat_size=kargs['cat_size'],
                               name= self.name + '.conv')

        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x, task_id=None, mask=None, mode="train"):
        return self.act(self.norm(self.conv(x, task_id, mask, mode)))


class HeadBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.name = kargs['name']
        ngf = kargs['ngf']
        bias = kargs['bias']
        sparsity = kargs['sparsity']
        self.conv = SubnetConv2d(ngf, 3, 1, 1, bias=bias, sparsity=sparsity)

    def forward(self, x, task_id=None, mask=None, mode="train"):

        if mask is not None:
            name = self.name + '.conv'
            weight_mask = mask[name + '.weight']
            bias_mask = mask[name + '.bias']
        else:
            weight_mask = None
            bias_mask = None

        out = self.conv(x, weight_mask=weight_mask, bias_mask=bias_mask, mode=mode)
        return out


class SubnetGenerator(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.name = 'generator'
        stem_dim, stem_num = [int(x) for x in kargs['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]
        mlp_dim_list = [kargs['embed_length'] * 2] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]

        self.sparsity = kargs['sparsity']
        self.device = kargs['device']
        self.stem = SubnetMLP(dim_list=mlp_dim_list, bias= kargs['bias'],
                              act=kargs['act'], sparsity=self.sparsity, name='stem')

        # BUILD CONV LAYERS
        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]
        ngf = self.fc_dim
        for i, stride in enumerate(kargs['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * kargs['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else kargs['reduction']), kargs['lower_width'])

            for j in range(kargs['num_blocks']):
                name = 'layers.{}'.format(i)
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                                            bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'],
                                            conv_type=kargs['conv_type'], sparsity=self.sparsity, 
                                            device=self.device, name=name))
                ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            name = 'head_layers.{}'.format(i)
            if kargs['sin_res']:
                if i == len(kargs['stride_list']) - 1:
                    head_layer = HeadBlock(ngf=ngf, bias=kargs['bias'], sparsity=self.sparsity, name=name)

                else:
                    head_layer = None
            else:
                head_layer = HeadBlock(ngf=ngf, bias=kargs['bias'], sparsity=self.sparsity, name=name)
            self.head_layers.append(head_layer)

        self.sigmoid = kargs['sigmoid']

    def get_masks(self):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask


    def forward(self, x, task_id=None, mask=None, mode="train"):

        if mode == 'test':
            per_task_mask = mask[task_id]
        else:
            per_task_mask = None

        output = self.stem(x, task_id=task_id, mask=per_task_mask, mode=mode)
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)

        out_list = []
        for layer, head_layer in zip(self.layers, self.head_layers):
            output = layer(output, task_id=task_id, mask=per_task_mask, mode=mode)
            if head_layer is not None:

                img_out = head_layer(output, task_id=task_id, mask=per_task_mask, mode=mode)

                # normalize the final output iwth sigmoid or tanh function
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)

        return  out_list



class SubnetGeneratorMH(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.name = 'generator'
        self.device = kargs['device']
        stem_dim, stem_num = [int(x) for x in kargs['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]
        mlp_dim_list = [kargs['embed_length'] * 2] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]

        assert kargs['lin_type'] in ['linear','freq4f', 'freq4s', 'freq4all']
        self.lin_type = kargs['lin_type']
        self.sparsity = kargs['sparsity']
        self.stem = SubnetMLP(dim_list=mlp_dim_list,
                              bias= kargs['bias'],
                              act=kargs['act'],
                              sparsity=self.sparsity,
                              device=self.device,
                              lin_type=self.lin_type,
                              name='stem')


        self.freq = True if kargs['freq'] >= 0 else False 

        # BUILD CONV LAYERS
        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]
        ngf = self.fc_dim
        for i, stride in enumerate(kargs['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * kargs['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else kargs['reduction']), kargs['lower_width'])

            for j in range(kargs['num_blocks']):
                name = 'layers.{}'.format(i)

                if i == kargs['freq'] and self.freq:

                    conv_type = 'convfreq_sum' if kargs['cat_size'] < 0 else 'convfreq_cat'
                    self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                                                 bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'],
                                                 conv_type=conv_type,
                                                 cat_size=kargs['cat_size'],
                                                 sparsity=self.sparsity,
                                                 device=self.device, name=name))
                else:
                    self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                                                 bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'],
                                                 conv_type=kargs['conv_type'],
                                                 cat_size=kargs['cat_size'],
                                                 sparsity=self.sparsity,
                                                 device=self.device, name=name))
                ngf = new_ngf

        # kargs['sin_res']:
        for t in range(kargs['n_tasks']):
            head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=kargs['bias'])
            self.head_layers.append(head_layer)

        self.sigmoid = kargs['sigmoid']

    def get_masks(self):
        task_mask = {}
        for name, module in self.named_modules():

            if 'head_layers' in name:
                continue

            # For the time being we only care about the current task outputhead
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d) or isinstance(module, SubnetConvTranspose2d):
                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

            elif isinstance(module, SpectralConv) or isinstance(module, SpectralLinear):
                task_mask[name + '.weight_real'] = [mask.detach().clone() if mask is not None else None for mask in module.weight_mask_real]

                task_mask[name + '.weight_imag'] = [mask.detach().clone() if mask is not None else None for mask in module.weight_mask_imag]

                #task_mask[name + '.weight_imag'] = None

                task_mask[name + '.bias'] = None

        return task_mask


    def reinit_masks(self):

        for name, module in self.named_modules():
            if 'head_layers' in name:
                continue

            # For the time being we only care about the current task outputhead
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d) or isinstance(module, SubnetConvTranspose2d):
                module.init_mask_parameters()

            elif isinstance(module, SpectralConv) or isinstance(module, SpectralLinear):
                module.init_mask_parameters()
        return


    def forward(self, x, task_id=None, mask=None, mode="train"):

        if mode == 'test':
            per_task_mask = mask[task_id]
        else:
            per_task_mask = None


        output = self.stem(x, task_id=task_id, mask=per_task_mask, mode=mode)
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)

        out_list = []
        for layer in self.layers:
            output = layer(output, task_id=task_id, mask=per_task_mask, mode=mode)

        img_out = self.head_layers[task_id](output)

        # normalize the final output iwth sigmoid or tanh function
        img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
        out_list.append(img_out)

        return  out_list
