#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Compute_Grad          import Compute_Grad

class model_GradUpdate3(torch.nn.Module):
    def __init__(self,ShapeData,GradType,dimState=30):
        super(model_GradUpdate3, self).__init__()

        with torch.no_grad():
            self.GradType  = GradType
            self.shape     = ShapeData
            self.DimState  = dimState
            self.layer_dim = 1
        self.compute_Grad  = Compute_Grad(ShapeData,GradType)
                
        if len(self.shape) == 2: ## 1D Data
            self.convLayer     = torch.Linear(dimState,self.shape[0]*self.shape[1])
            self.lstm = torch.nn.LSTM(self.shape[0]*self.shape[1],self.DimState,self.layer_dim)
        else:
            self.convLayer     = torch.Linear(dimState,self.shape[0]*self.shape[1]*self.shape[2])
            self.lstm = torch.nn.LSTM(self.shape[0]*self.shape[1]*self.shape[2],self.DimState,self.layer_dim)
    def forward(self, x,xpred,xobs,mask,hidden,cell,gradnorm=1.0):

        # compute gradient
        grad = self.compute_Grad(x, xpred,xobs,mask)
        
        #grad = grad /self.ScaleGrad
        #grad = grad / torch.sqrt( torch.mean( grad**2 ) )
        #grad = self.bn1(grad)
        grad  = grad / gradnorm

                     
        if len(self.shape) == 2: ## 1D Data
            grad = grad.view(-1,1,self.shape[0]*self.shape[1])
        else:
            grad = grad.view(-1,1,self.shape[0]*self.shape[1]*self.shape[2])
        
        if hidden is None:
            output,(hidden,cell)  = self.lstm(grad,None)
        else:
            output,(hidden,cell) = self.lstm(grad,(hidden,cell))

        grad = self.convLayer( output )
        if len(self.shape) == 2: ## 1D Data
            grad = grad.view(-1,self.shape[0],self.shape[1])
        else:
            grad = grad.view(-1,self.shape[0],self.shape[1],self.shape[2])

        return grad,hidden,cell

