#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .mods.utils.utils_solver.model_GradUpdate0    import model_GradUpdate0
from .mods.utils.utils_solver.model_GradUpdate1    import model_GradUpdate1
from .mods.utils.utils_solver.model_GradUpdate2    import model_GradUpdate2
from .mods.utils.utils_solver.model_GradUpdate3    import model_GradUpdate3

def replace_tup_at_idx(tup, idx, val):
    lst = list(tup)
    lst[idx] = val
    return tuple(lst)

# NN architecture based on given dynamical NN prior (model_AE)
# solving the reconstruction of the hidden states using a number of 
# fixed-point iterations prior to a gradient-based minimization
class Model_4DVarNN_Grad(torch.nn.Module):
    def __init__(self,mod_AE,ShapeData,NiterGrad,GradType,OptimType,InterpFlag=False,periodicBnd=False,N_cov=0):
    #def __init__(self,mod_AE,GradType,OptimType):
        super(Model_4DVarNN_Grad, self).__init__()
        self.model_AE    = mod_AE
        self.OptimType   = OptimType
        self.GradType    = GradType
        self.Ncov        = N_cov
        self.InterpFlag  = InterpFlag
        self.periodicBnd = periodicBnd
        self.shape       = ShapeData
        # Define Solver type according to OptimType
        ## Gradient-based minimization using a fixed-step descent
        if OptimType == 0:
          self.model_Grad = model_GradUpdate0(replace_tup_at_idx(self.shape,0,int(self.shape[0]/(self.Ncov+1))),GradType)
        ## Gradient-based minimization using a CNN using a (sub)gradient as inputs 
        elif OptimType == 1:
          self.model_Grad = model_GradUpdate1(replace_tup_at_idx(self.shape,0,int(self.shape[0]/(self.Ncov+1))),GradType,self.periodicBnd)
        ## Gradient-based minimization using a LSTM using a (sub)gradient as inputs
        elif OptimType == 2:
          self.model_Grad = model_GradUpdate2(replace_tup_at_idx(self.shape,0,int(self.shape[0]/(self.Ncov+1))),GradType,self.periodicBnd)
        elif OptimType == 3:
          self.model_Grad = model_GradUpdate3(replace_tup_at_idx(self.shape,0,int(self.shape[0]/(self.Ncov+1))),GradType,30)
        elif OptimType == 4:
          self.model_Grad = model_GradUpdate4(replace_tup_at_idx(self.shape,0,int(self.shape[0]/(self.Ncov+1))),GradType,self.periodicBnd)
                    
        with torch.no_grad():
            self.OptimType = OptimType
            self.NGrad     = int(NiterGrad)
            
    def forward(self, x_inp,xobs,mask,g1=None,g2=None,normgrad=0.):
        if( self.InterpFlag == True ):
            #mask   = torch.add(1.0,torch.mul(mask,0.0)) # set mask to 1 # debug
            mask_  = torch.add(1.0,torch.mul(mask,-1.0)) #1. - mask
        x      = torch.mul(x_inp,1.0)

        # new index to select appropriate data if covariates are used
        index = np.arange(0,self.shape[0],self.Ncov+1)  

        # gradient iteration
        for kk in range(0,self.NGrad):
            # gradient normalisation
            grad     = self.model_Grad.compute_Grad(x[index], self.model_AE(x),xobs,mask)
            if normgrad == 0. :
                _normgrad = torch.sqrt( torch.mean( grad**2 ) )
            else:
                _normgrad = normgrad
            # AE pediction
            xpred = self.model_AE(x)
            # gradient update
            ## Gradient-based minimization using a fixed-step descent
            if self.OptimType == 0:
                grad  = self.model_Grad( x[index], xpred, xobs, mask[index], _normgrad )
            ## Gradient-based minimization using a CNN using a (sub)gradient as inputs
            elif self.OptimType == 1:               
              if kk == 0:
                grad  = self.model_Grad( x[index], xpred, xobs, mask[index], g1, _normgrad )
              else:
                grad  = self.model_Grad( x[index], xpred, xobs, mask[index],grad_old, _normgrad )
              grad_old = torch.mul(1.,grad)
            ## Gradient-based minimization using a LSTM using a (sub)gradient as inputs    
            elif self.OptimType == 2:               
              if kk == 0:
                grad,hidden,cell  = self.model_Grad( x[index], xpred, xobs, mask[index], g1, g2, _normgrad )
              else:
                grad,hidden,cell  = self.model_Grad( x[index], xpred, xobs, mask[index], hidden, cell, _normgrad )
            # interpolation constraint
            if( self.InterpFlag == True ):
                # optimization update
                xnew = x[index] - grad
                x[index]    = x[index] * mask[index] + xnew * mask_[index]
            else:
                # optimization update
                x[index] = x[index] - grad
        if self.OptimType == 1:
            return x,grad_old,_normgrad
        if self.OptimType == 2:
            return x,hidden,cell,_normgrad
        else:
            return x,_normgrad
