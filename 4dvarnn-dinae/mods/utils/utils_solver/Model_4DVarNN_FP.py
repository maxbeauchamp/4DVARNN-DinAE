#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NN architecture based on given dynamical NN prior (model_AE)
# solving the reconstruction of the hidden states using fixed-point iterations
class Model_4DVarNN_FP(torch.nn.Module):
    def __init__(self,mod_AE,ShapeData,NiterProjection):
    #def __init__(self,mod_AE,GradType,OptimType):
        super(Model_4DVarNN_FP, self).__init__()
        self.model_AE = mod_AE
    
        with torch.no_grad():
            self.NProjFP   = int(NiterProjection)
            
    def forward(self, x_inp,xobs,mask,g1=None,g2=None):
        mask_  = torch.add(1.0,torch.mul(mask,-1.0)) #1. - mask
        
        x      = torch.mul(x_inp,1.0)

        # fixed-point iterations
        for kk in range(0,self.NProjFP):
            x_proj = self.model_AE(x)
            x_proj = torch.mul(x_proj,mask_)
            x      = torch.mul(x, mask)   
            x      = torch.add(x , x_proj )

        return x

