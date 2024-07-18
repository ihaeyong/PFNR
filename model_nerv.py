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

class CustomDataSetMTL1(Dataset):
    def __init__(self, main_dir_list, transform, vid_list=[None], frame_gap=1):

        self.frame_path = []
        self.task_id = []
        self.frame_idx = []
        for id, main_dir in enumerate(main_dir_list):
            # import ipdb; ipdb.set_trace()
            print('loading...'+ main_dir)
            #self.main_dir = main_dir
            self.transform = transform
            frame_idx = []
            accum_img_num = []
            all_imgs = os.listdir(main_dir)
            all_imgs.sort()

            num_frame = 0
            for img_id in all_imgs:
                img_id = os.path.join(main_dir, img_id)
                self.frame_path.append(img_id)
                frame_idx.append(num_frame)
                self.task_id.append(id)
                num_frame += 1

            accum_img_num.append(num_frame)

            frame_idx = [float(x) / len(frame_idx) for x in frame_idx]
            self.accum_img_num = np.asfarray(accum_img_num)
            if None not in vid_list:
                frame_idx = [frame_idx[i] for i in vid_list]
            
            # import ipdb; ipdb.set_trace()
            self.frame_idx.extend(frame_idx)
            
            self.frame_gap = frame_gap

    def __len__(self):
        return len(self.frame_idx) // self.frame_gap

    def __getitem__(self, idx):
        valid_idx = idx * self.frame_gap
        img_name = self.frame_path[valid_idx]
        #img_name = os.path.join(self.main_dir, img_id)
        image = Image.open(img_name).convert("RGB")
        tensor_image = self.transform(image)
        if tensor_image.size(1) > tensor_image.size(2):
            tensor_image = tensor_image.permute(0,2,1)

        tensor_image = F.adaptive_avg_pool2d(tensor_image, (720, 1280))

        frame_idx = torch.tensor(self.frame_idx[valid_idx])
        task_id = torch.tensor(self.task_id[valid_idx])

        return tensor_image, frame_idx, task_id
    
class CustomDataSetMTL(Dataset):
    def __init__(self, data, norm_idx, task_id, max_num_tasks):
        self.data = data
        self.norm_idx = norm_idx
        self.task_id = task_id
        self.max_num_tasks = max_num_tasks

    def __len__(self):
        return self.max_num_tasks

    def __getitem__(self, idx):

        data = self.data[idx]
        norm_idx = self.norm_idx[idx]
        task_id = self.task_id[idx]
        
        return data, norm_idx, task_id

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
        self.subnet= kargs['subnet']
        self.sparsity = kargs['sparsity']
        self.conv_type = kargs['conv_type']
        self.name = kargs['name']

        if self.conv_type == 'conv':
            if not self.subnet:
                self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
            else:
                self.conv = SubnetConv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'], sparsity=self.sparsity)

            self.up_scale = nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            if not self.subnet:
                self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
            else:
                self.conv = SubnetConvTranspose2d(ngf, new_ngf, stride, stride, sparsity=self.sparsity)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            if not self.subnet:
                self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
                self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])
            else:
                self.conv = None
                self.up_scale = None


        elif self.conv_type == 'convfreq_sum' :

            self.device = kargs['device']
            self.scale = 0.1
            self.var = 0.5
            self.idx_th = 0

            if self.name == 'layers.0.conv':
                n_modes = (9,16)
            elif self.name == 'layers.1.conv':
                n_modes = (45,80)
            elif self.name == 'layers.2.conv':
                n_modes = (90,160)
            elif self.name == 'layers.3.conv':
                n_modes = (180,320)

            self.conv = nn.Conv2d(ngf,
                                  new_ngf * stride * stride, 3, 1, 1,
                                  bias=kargs['bias'])

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
            # fft_sparsity = 1 - (1-self.sparsity) * self.fft_scale
            fft_sparsity = 0.5 # dense

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

    def forward(self, x):

        if self.conv_type == 'convfreq_sum':

            out = self.conv(x)
            out = self.up_scale(out)

            # mask is none since it is a dense layer
            weight_mask_real = None
            weight_mask_imag = None

            out += self.conv_freq(x=x,indices=0, output_shape=None,
                                  weight_mask_real=weight_mask_real,
                                  weight_mask_imag=weight_mask_imag,
                                  mode='train')

            return out
        else:

            out = self.conv(x)
            return self.up_scale(out)


# Multiple Input Sequential
class Sequential(nn.Sequential):

    def forward(self, *inputs):
        inputs = inputs[0]
        mask = inputs[1]
        mode = inputs[2]

        for module in self._modules.values():
            if isinstance(module, SubnetMLP):
                inputs = module(inputs, mask, mode)
            else:
                inputs = module(inputs)

        return inputs


def MLP(dim_list, act='relu', bias=True):
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias), act_fn]
    return nn.Sequential(*fc_list)


def SubnetMLP(dim_list, act='relu', bias=True, sparsity=0.3, name='mlp'):
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_name = name + '.' + str(i)
        fc_list += [SubnetLinear(dim_list[i], dim_list[i+1], bias=bias,
                                 sparsity=sparsity), act_fn]
    return Sequential(*fc_list)


class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.name = 'name'
        self.conv = CustomConv(ngf=kargs['ngf'],
                               new_ngf=kargs['new_ngf'],
                               stride=kargs['stride'],
                               bias=kargs['bias'],
                               conv_type=kargs['conv_type'],
                               subnet=kargs['subnet'],
                               sparsity=kargs['sparsity'],
                               device=kargs['device'],
                               name=kargs['name'] + '.conv')

        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.name = 'generator'
        stem_dim, stem_num = [int(x) for x in kargs['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]
        mlp_dim_list = [kargs['embed_length']] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]

        self.subnet = kargs['subnet']
        self.sparsity = kargs['sparsity']
        if not self.subnet:
            self.stem = MLP(dim_list=mlp_dim_list, act=kargs['act'])
        else:
            self.stem = SubnetMLP(dim_list=mlp_dim_list, act=kargs['act'], sparsity=self.sparsity)

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
                                                 conv_type=conv_type, subnet=self.subnet,
                                                 sparsity=self.sparsity, device=kargs['device'],
                                                 name=name))

                else:
                    self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                                                 bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'],
                                                 conv_type=kargs['conv_type'], subnet=self.subnet,
                                                 sparsity=self.sparsity, device=kargs['device'],
                                                 name=name))



                ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if kargs['sin_res']:
                if i == len(kargs['stride_list']) - 1:
                    if self.subnet:
                        head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=kargs['bias']) 
                    else:
                        head_layer = SubnetConv2d(ngf, 3, 1, 1, bias=kargs['bias'], sparsity=self.sparsity)

                else:
                    head_layer = None
            else:
                if not self.subnet:
                    head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=kargs['bias'])
                else:
                    head_layer = SubnetConv2d(ngf, 3, 1, 1, bias=kargs['bias'], sparsity=self.sparsity)
            self.head_layers.append(head_layer)
        self.sigmoid =kargs['sigmoid']

    def forward(self, input):
        output = self.stem(input)
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)

        out_list = []
        for layer, head_layer in zip(self.layers, self.head_layers):
            output = layer(output)
            if head_layer is not None:
                img_out = head_layer(output)

                # normalize the final output iwth sigmoid or tanh function
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)

        return  out_list
    




class GeneratorMTL(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.name = 'generator'
        stem_dim, stem_num = [int(x) for x in kargs['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')]
        mlp_dim_list = [kargs['embed_length'] * 2] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]

        self.subnet = kargs['subnet']
        self.sparsity = kargs['sparsity']
        if not self.subnet:
            self.stem = MLP(dim_list=mlp_dim_list, act=kargs['act'])
        else:
            self.stem = SubnetMLP(dim_list=mlp_dim_list, act=kargs['act'], sparsity=self.sparsity)

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
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                                             bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'], conv_type=kargs['conv_type'], subnet=self.subnet, sparsity=self.sparsity))
                ngf = new_ngf


        # kargs['sin_res']:
        for t in range(kargs['n_tasks']):
            head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=kargs['bias'])
            self.head_layers.append(head_layer)

        self.sigmoid =kargs['sigmoid']

    def forward(self, input, task_id=None):
        output = self.stem(input)
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)

        out_list = []
        for layer in self.layers:
            output = layer(output)

        img_out = []
        for i, id in enumerate(task_id):
            out = self.head_layers[task_id[i]](output[i])
            img_out.append(out[None])
        
        img_out = torch.cat(img_out)

        # normalize the final output iwth sigmoid or tanh function
        img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
        out_list.append(img_out)

        return out_list

