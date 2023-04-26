import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
import math

import numpy as np

def percentile(scores, sparsity):
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()

class GetSubnetFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):

        #scores += torch.randn_like(scores) * 1e-16
        k_val = percentile(scores, sparsity * 100)
        masks = torch.where(scores < k_val,
                            zeros.to(scores.device),
                            ones.to(scores.device))
        return masks

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None

## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class SubnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.1, trainable=True):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weights and Bias
        self.w_m = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        if bias:
            self.b_m = nn.Parameter(torch.empty(out_features))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters()
        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

    def forward(self, x, weight_mask=None, bias_mask=None, mode='train'):

        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if mode=='train':
            if weight_mask is None:
                self.weight_mask=GetSubnetFaster.apply(self.w_m.abs(),
                                                       self.zeros_weight,
                                                       self.ones_weight,
                                                       self.sparsity)
            else:
                self.weight_mask = weight_mask
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(),
                                                       self.zeros_bias,
                                                       self.ones_bias,
                                                       self.sparsity)
                b_pruned = self.bias_mask * self.bias

        elif mode=='valid':
            w_pruned = self.weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

        # If inference, no need to compute the subnetwork
        elif mode=='test':
            w_pruned = weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def update_w_m(self):
        self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                 self.zeros_weight,
                                                 self.ones_weight,
                                                 self.sparsity)

    def init_mask_parameters(self, uniform=True):
        if uniform:
            nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        else:
            nn.init.normal_(self.w_m, std)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)

class SubnetConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, sparsity=0.1, trainable=True):

        super(self.__class__, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.stride = stride
        # self.padding = padding
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weight and Bias
        self.w_m = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)

        if bias:
            self.b_m = nn.Parameter(torch.empty(out_channels))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters()

        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

    def forward(self, x, weight_mask=None, bias_mask=None, mode='train'):

        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if mode == 'train':
            self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                     self.zeros_weight,
                                                     self.ones_weight,
                                                     self.sparsity)
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(), self.zeros_bias, self.ones_bias, self.sparsity)
                b_pruned = self.bias_mask * self.bias

        elif mode=='valid':
            w_pruned = self.weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

        # If inference, no need to compute the subnetwork
        elif mode == 'test':
            w_pruned = weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        return F.conv2d(input=x, weight=w_pruned, bias=b_pruned, stride=self.stride, padding=self.padding)

    def update_w_m(self):
        self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                 self.zeros_weight,
                                                 self.ones_weight,
                                                 self.sparsity,
                                                 self.soft)

    def init_mask_parameters(self, uniform=True):
        if uniform:
            nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        else:
            nn.init.normal_(self.w_m, std)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)


class SubnetConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, sparsity=0.1, trainable=True):

        super(self.__class__, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.stride = stride
        # self.padding = padding
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weight and Bias
        self.w_m = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)

        if bias:
            self.b_m = nn.Parameter(torch.empty(in_channels))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters()

        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

    def forward(self, x, weight_mask=None, bias_mask=None, mode='train'):

        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if mode == 'train':
            self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                     self.zeros_weight,
                                                     self.ones_weight,
                                                     self.sparsity)
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(), self.zeros_bias, self.ones_bias, self.sparsity)
                b_pruned = self.bias_mask * self.bias

        elif mode=='valid':
            w_pruned = self.weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

        # If inference, no need to compute the subnetwork
        elif mode == 'test':
            w_pruned = weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        return F.conv_transpose2d(input=x, weight=w_pruned, bias=b_pruned, stride=self.stride, padding=self.padding)

    def update_w_m(self):
        self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                 self.zeros_weight,
                                                 self.ones_weight,
                                                 self.sparsity,
                                                 self.soft)

    def init_mask_parameters(self, uniform=True):
        if uniform:
            nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        else:
            nn.init.normal_(self.w_m, std)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)
