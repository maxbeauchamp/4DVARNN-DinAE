#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .mods.utils.utils_solver.ComputeGrad          import ComputeGrad

# Gradient-based minimization using a fixed-step descent
class model_GradUpdate0(torch.nn.Module):
    def __init__(self,ShapeData,GradType):
        super(model_GradUpdate0, self).__init__()

        with torch.no_grad():
            self.GradType  = GradType
            self.shape     = ShapeData
            self.delta     = torch.nn.Parameter(torch.Tensor([1.]))
        self.compute_Grad  = Compute_Grad(ShapeData,GradType)
        self.gradNet       = self._make_ConvGrad()
        self.bn1           = torch.nn.BatchNorm2d(self.shape[0])
        
    def _make_ConvGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(torch.nn.Conv1d(self.shape[0], self.shape[0],1, padding=0,bias=False))
        elif len(self.shape) == 3: ## 2D Data
            conv1 = torch.nn.Conv2d(self.shape[0], self.shape[0], (1,1), padding=0,bias=False)      
            # predefined parameters
            K = torch.Tensor([0.1]).view(1,1,1,1) # should be 0.1 is no bn is used
            conv1.weight = torch.nn.Parameter(K)
            layers.append(conv1)            

        return torch.nn.Sequential(*layers)

    def forward(self, x,xpred,xobs,mask,gradnorm=1.0):

        # compute gradient
        grad = self.compute_Grad(x, xpred,xobs,mask)
        
        #print( torch.sqrt( torch.mean( grad**2 ) ) )
        
        # scaling gradient values
        #grad = grad /self.ScaleGrad
        #grad = grad / torch.sqrt( torch.mean( grad**2 ) )
        
        # update
        grad = self.gradNet( grad )
        #grad = self.gradNet( self.bn1( grad ) )

        return grad
