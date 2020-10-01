#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

from dinae_4dvarnn import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Residual Block architecture (Conv2d)
class ResidualBlock(torch.nn.Module):
    def __init__(self, dim,K,
                 kernel_size,
                 padding=0):
        super(ResidualBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(dim,K*dim,kernel_size,padding=padding,bias=False)
        self.conv2 = torch.nn.Conv2d(K*dim,dim,kernel_size,padding=padding,bias=False)

    def forward(self, x):

        dx = self.conv1( F.relu(x) )
        dx_lin = self.conv2(dx)
        dx1 = self.conv2(dx)
        dx2 = self.conv2(dx)
        dx1 = torch.mul(dx1,dx2)
        dx  = torch.add(dx1,dx_lin)
        dx  = torch.tanh(dx)
        x   = torch.add(x,dx)

        return x


## ResNet architecture (Conv2d)      
class ResNetConv2d(torch.nn.Module):
  def __init__(self,Nblocks,dim,K,
                 kernel_size,
                 padding=0):
      super(ResNetConv2d, self).__init__()
      self.resnet = self._make_ResNet(Nblocks,dim,K,kernel_size,padding)

  def _make_ResNet(self,Nblocks,dim,K,kernel_size,padding):
      layers = []
      for kk in range(0,Nblocks):
        layers.append(ResidualBlock(dim,K,kernel_size,padding))

      return torch.nn.Sequential(*layers)

  def forward(self, x):
      x = self.resnet ( x )

      return x
