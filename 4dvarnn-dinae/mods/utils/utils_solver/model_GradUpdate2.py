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
from .mods.utils.utils_solver.ConvLSTM1d           import ConvLSTM1d
from .mods.utils.utils_solver.ConvLSTM2d           import ConvLSTM2d

# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class model_GradUpdate2(torch.nn.Module):
    def __init__(self,ShapeData,GradType,periodicBnd=False):
        super(model_GradUpdate2, self).__init__()
        with torch.no_grad():
            self.GradType  = GradType
            self.shape     = ShapeData
            self.DimState  = 5*self.shape[0]
            self.PeriodicBnd = periodicBnd
            if( (self.PeriodicBnd == True) & (len(self.shape) == 2) ):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False
        self.compute_Grad  = Compute_Grad(ShapeData,GradType)
        self.convLayer     = self._make_ConvGrad()
        #self.bn1           = torch.nn.BatchNorm2d(self.shape[0])
        #self.lstm            = self._make_LSTMGrad()
        K = torch.Tensor([0.1]).view(1,1,1,1)
        self.convLayer.weight = torch.nn.Parameter(K)
        if len(self.shape) == 2: ## 1D Data
            self.lstm = ConvLSTM1d(self.shape[0],self.DimState,3)
        elif len(self.shape) == 3: ## 2D Data
            self.lstm = ConvLSTM2d(self.shape[0],self.DimState,3)

    def _make_ConvGrad(self):
        layers = []
        if len(self.shape) == 2: ## 1D Data
            layers.append(torch.nn.Conv1d(5*self.shape[0], self.shape[0], 1, padding=0,bias=False))
        elif len(self.shape) == 3: ## 2D Data            
            layers.append(torch.nn.Conv2d(5*self.shape[0], self.shape[0], (1,1), padding=0,bias=False))
        return torch.nn.Sequential(*layers)

    def _make_LSTMGrad(self):
        layers = []
        if len(self.shape) == 2: ## 1D Data
            layers.append(ConvLSTM1d(self.shape[0],5*self.shape[0],3))
        elif len(self.shape) == 3: ## 2D Data            
            layers.append(ConvLSTM2d(self.shape[0],5*self.shape[0],3))
        return torch.nn.Sequential(*layers)
 
    def forward(self, x,xpred,xobs,mask,hidden,cell,gradnorm=1.0):

        # compute gradient
        grad = self.compute_Grad(x, xpred,xobs,mask)
        #grad = grad /self.ScaleGrad
        #grad = grad / torch.sqrt( torch.mean( grad**2 ) )
        #grad = self.bn1(grad)
        grad  = grad / gradnorm
        if self.PeriodicBnd == True :
            dB     = 7
            grad_  = torch.cat((grad[:,:,x.size(2)-dB:,:],grad,grad[:,:,0:dB,:]),dim=2)
            if hidden is None:
                hidden_,cell_ = self.lstm(grad_,None)
            else:
                hidden_  = torch.cat((hidden[:,:,x.size(2)-dB:,:],hidden,hidden[:,:,0:dB,:]),dim=2)
                cell_    = torch.cat((cell[:,:,x.size(2)-dB:,:],cell,cell[:,:,0:dB,:]),dim=2)
                hidden_,cell_ = self.lstm(grad_,[hidden_,cell_])
            hidden = hidden_[:,:,dB:x.size(2)+dB,:]
            cell   = cell_[:,:,dB:x.size(2)+dB,:]
        else:
            if hidden is None:
                hidden,cell = self.lstm(grad,None)
            else:
                hidden,cell = self.lstm(grad,[hidden,cell])
        grad = self.convLayer( hidden )

        return grad,hidden,cell

