#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def replace_tup_at_idx(tup, idx, val):
    lst = list(tup)
    lst[idx] = val
    return tuple(lst)

# NN architecture based on given dynamical NN prior (model_AE)
# solving the reconstruction of the hidden states using fixed-point iterations
class Model_4DVarNN_FP(torch.nn.Module):
    def __init__(self,mod_AE,ShapeData,NiterProjection,N_cov=0):
    #def __init__(self,mod_AE,GradType,OptimType):
        super(Model_4DVarNN_FP, self).__init__()
        self.model_AE = mod_AE
        self.Ncov     = N_cov
        self.shape       = ShapeData
        with torch.no_grad():
            self.NProjFP   = int(NiterProjection)
            
    def forward(self, x_inp,xobs,mask,g1=None,g2=None):
        mask_  = torch.add(1.0,torch.mul(mask,-1.0)) #1. - mask
        x      = torch.mul(x_inp,1.0)

        # new index to select appropriate data if covariates are used
        index = np.arange(0,self.shape[0],self.Ncov+1)  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_index = torch.LongTensor(np.float64(index)).to(device)
        with torch.set_grad_enabled(True), torch.autograd.set_detect_anomaly(True):
            target_x     = x[:,index,:,:]
            target_obs   = xobs[:,index,:,:]
            target_mask  = mask[:,index,:,:]
            target_mask_ = mask_[:,index,:,:]

        # fixed-point iterations
        for kk in range(0,self.NProjFP):
            x_proj   = self.model_AE(x)
            x_proj   = torch.mul(x_proj,target_mask_)
            target_x = torch.add(torch.mul(target_x,target_mask),x_proj)
            x = torch.Tensor.index_fill(x,1,torch_index,0)
            x = torch.Tensor.index_add(x,1,torch_index,target_x)

        return target_x

