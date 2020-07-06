#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constrained Conv2D Layer with zero-weight at central point
class ConstrainedConv2d(torch.nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(ConstrainedConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias)
        with torch.no_grad():
          self.weight[:,:,int(self.weight.size(2)/2)+1,int(self.weight.size(3)/2)+1] = 0.0
    def forward(self, input):
        return torch.nn.functional.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

