#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constrained Conv1D Layer with zero-weight at central point
class ConstrainedConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(ConstrainedConv1d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias)
        with torch.no_grad():
          self.weight[:,:,int(self.weight.size(2)/2)+1] = 0.0
    def forward(self, input):
        return torch.nn.functional.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
