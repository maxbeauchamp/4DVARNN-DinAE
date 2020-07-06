#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## ResNet architecture (Conv2d)      
class ResNetConv2d(torch.nn.Module):
  def __init__(self,Nblocks,dim,K,
                 kernel_size,
                 padding=0):
      super(ResNetConv2D, self).__init__()
      self.resnet = self._make_ResNet(Nblocks,dim,K,kernel_size,padding)

  def _make_ResNet(self,Nblocks,dim,K,kernel_size,padding):
      layers = []
      for kk in range(0,Nblocks):
        layers.append(torch.nn.Conv2d(dim,K*dim,kernel_size,padding=padding,bias=False))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(K*dim,dim,kernel_size,padding=padding,bias=False))

      return torch.nn.Sequential(*layers)

  def forward(self, x):
      x = self.resnet ( x )

      return x
