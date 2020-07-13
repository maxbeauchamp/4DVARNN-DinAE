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

# Gradient-based minimization using a CNN using a (sub)gradient as inputs
class model_GradUpdate1(torch.nn.Module):
    def __init__(self,ShapeData,GradType,periodicBnd=False):
        super(model_GradUpdate1, self).__init__()

        with torch.no_grad():
            self.GradType  = GradType
            self.shape     = ShapeData
            #self.delta     = torch.nn.Parameter(torch.Tensor([1e4]))
            self.PeriodicBnd = periodicBnd
            
            if( (self.PeriodicBnd == True) & (len(self.shape) == 2) ):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False
        self.compute_Grad  = Compute_Grad(ShapeData,GradType)
        self.gradNet1      = self._make_ConvGrad()
                
        if len(self.shape) == 2: ## 1D Data
            self.gradNet2      = torch.nn.Conv1d(self.shape[0], self.shape[0], 1, padding=0,bias=False)
            self.gradNet3      = torch.nn.Conv1d(self.shape[0], self.shape[0], 1, padding=0,bias=False)
            print( self.gradNet3.weight.size() )
            K = torch.Tensor(np.identity( self.shape[0] )).view(self.shape[0],self.shape[0],1)
            self.gradNet3.weight = torch.nn.Parameter(K)
        elif len(self.shape) == 3: ## 2D Data            
            self.gradNet2      = torch.nn.Conv2d(self.shape[0], self.shape[0], (1,1), padding=0,bias=False)
            self.gradNet3      = torch.nn.Conv2d(self.shape[0], self.shape[0], (1,1), padding=0,bias=False)
        #self.bn1           = torch.nn.BatchNorm2d(self.shape[0])
        #self.bn2           = torch.nn.BatchNorm2d(self.shape[0])
            K = torch.Tensor(np.identity( self.shape[0] )).view(self.shape[0],self.shape[0],1,1)
            self.gradNet3.weight = torch.nn.Parameter(K)

        #with torch.enable_grad():
        #with torch.enable_grad(): 

    def _make_ConvGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(torch.nn.Conv1d(2*self.shape[0], 8*self.shape[0],3, padding=1,bias=False))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Conv1d(8*self.shape[0], 16*self.shape[0],3, padding=1,bias=False))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Conv1d(16*self.shape[0], self.shape[0],3, padding=1,bias=False))
        elif len(self.shape) == 3: ## 2D Data            
            #layers.append(torch.nn.Conv2d(2*self.shape[0], self.shape[0], (3,3), padding=1,bias=False))
            layers.append(torch.nn.Conv2d(2*self.shape[0], 8*self.shape[0], (3,3), padding=1,bias=False))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Conv2d(8*self.shape[0], 16*self.shape[0], (3,3), padding=1,bias=False))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Conv2d(16*self.shape[0], self.shape[0], (3,3), padding=1,bias=False))

        return torch.nn.Sequential(*layers)
 
    def forward(self, x,xpred,xobs,mask,grad_old,gradnorm=1.0):

        # compute gradient
        grad = self.compute_Grad(x, xpred,xobs,mask)
         
        #grad = grad /self.ScaleGrad
        #grad = grad / torch.sqrt( torch.mean( grad**2 ) )
        grad  = grad / gradnorm
        
        if grad_old is None:
            grad_old = torch.randn(grad.size()).to(device) ## Here device is global variable to be checked
 
        # boundary conditons
        if self.PeriodicBnd == True :
            dB     = 7
            #
            grad_  = torch.cat((grad[:,:,x.size(2)-dB:,:],grad,grad[:,:,0:dB,:]),dim=2)
                
            grad_old_ = torch.cat((grad_old[:,:,x.size(2)-dB:,:],grad_old,grad_old[:,:,0:dB,:]),dim=2)
    
            gradAll   = torch.cat((grad_old_,grad_),1)
    
            dgrad = self.gradNet1( gradAll )
            grad = grad + self.gradNet2( dgrad[:,:,dB:x.size(2)+dB,:] )
            
            #dgrad = self.gradNet2( self.bn1(dgrad[:,:,dB:x.size(2)+dB,:] ) )
            #grad  = self.bn2(grad) +  dgrad
        else:
            gradAll   = torch.cat((grad_old,grad),1)        
    
            dgrad = self.gradNet1( gradAll )
            grad  = grad + self.gradNet2( dgrad )

        grad      = 5. * torch.atan( 0.2 * self.gradNet3( grad ) )
        
        return grad

