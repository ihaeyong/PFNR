import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from model.subnet import SubnetConv2d, SubnetLinear, SubnetConvTranspose2d


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
        self.subnet= kargs['subnet']
        self.sparsity = kargs['sparsity']
        self.conv_type = kargs['conv_type']
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

    def forward(self, x):
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
                               sparsity=kargs['sparsity'])

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

        import ipdb; ipdb.set_trace()
        for layer, head_layer in zip(self.layers, self.head_layers):
            output = layer(output)
            if head_layer is not None:
                img_out = head_layer(output)

                # normalize the final output iwth sigmoid or tanh function
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)

        return  out_list

