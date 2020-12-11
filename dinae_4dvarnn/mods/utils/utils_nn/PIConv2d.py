#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Physics informed Conv2D Layer deriving from spatial advection-diffusion equation
class PIConv2d(torch.nn.Conv2d):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 padding=int(3/2),
                 dilation=1,
                 groups=1,
                 bias=True):
        super(PIConv2d,
              self).__init__(in_channels, in_channels, kernel_size, stride,
                             padding, dilation, groups, bias)
        with torch.no_grad():
            self.weight[:,:,0,0] = self.weight[:,:,2,2]
            self.weight[:,:,0,1] = self.weight[:,:,2,1]
            self.weight[:,:,0,2] = -1.*self.weight[:,:,0,0]
            self.weight[:,:,1,0] = self.weight[:,:,1,2]
            self.weight[:,:,2,0] = -1*self.weight[:,:,0,0]
            self.weight[:,:,2,0] = -1*self.weight[:,:,0,0]
            # central point
            self.weight[:,:,1,1] = -2*(self.weight[:,:,0,1]+self.weight[:,:,1,0]) #+ K2

    def forward(self, input):
        return torch.nn.functional.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

