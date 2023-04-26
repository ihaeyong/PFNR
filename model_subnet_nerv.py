# Authored by Haeyong Kang
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
        self.name = kargs['name']
        self.sparsity = kargs['sparsity']
        self.conv_type = kargs['conv_type']

        if self.conv_type == 'conv':
            self.conv = SubnetConv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'], sparsity=self.sparsity)
            self.up_scale = nn.PixelShuffle(stride)

        elif self.conv_type == 'deconv':
            self.conv = SubnetConvTranspose2d(ngf, new_ngf, stride, stride, sparsity=self.sparsity)
            self.up_scale = nn.Identity()

        elif self.conv_type == 'bilinear':
            if not self.subnet:
                self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
                self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])
            else:
                self.conv = None
                self.up_scale = None

    def forward(self, x, task_id=None, mask=None, mode="train"):

        if mask is not None:
            name = self.name + '.conv'
            weight_mask = mask[name + '.weight']
            bias_mask = mask[name + '.bias']
        else:
            weight_mask = None
            bias_mask = None

        out = self.conv(x, weight_mask=weight_mask, bias_mask=bias_mask, mode=mode)
        return self.up_scale(out)

class SubnetMLP(nn.Module):
    def __init__(self, **kargs):
        super(SubnetMLP, self).__init__()

        self.name = kargs['name']
        self.act_fn = ActivationLayer('relu')
        self.dim_list = kargs['dim_list']
        self.bias = kargs['bias']
        self.sparsity = kargs['sparsity']

        assert len(self.dim_list)-1 == 2

        self.mlp_fc1 = SubnetLinear(self.dim_list[0], self.dim_list[1],
                                    bias=self.bias, sparsity=self.sparsity)

        self.mlp_fc2 = SubnetLinear(self.dim_list[1], self.dim_list[2],
                                    bias=self.bias, sparsity=self.sparsity)

    def forward(self, x, task_id=None, mask=None, mode="train"):

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
        mlp_dim_list = [kargs['embed_length']] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]

        self.sparsity = kargs['sparsity']
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
                                             conv_type=kargs['conv_type'], sparsity=self.sparsity, name=name))
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

        self.sigmoid =kargs['sigmoid']

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

