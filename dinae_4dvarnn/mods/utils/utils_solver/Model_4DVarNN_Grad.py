#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model_GradUpdate0    import model_GradUpdate0
from model_GradUpdate1    import model_GradUpdate1
from model_GradUpdate2    import model_GradUpdate2
from model_GradUpdate3    import model_GradUpdate3

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_index = torch.LongTensor(np.float64(index)).to(device)
        with torch.set_grad_enabled(True), torch.autograd.set_detect_anomaly(True):
            target_x     = x[:,index,:,:]
            target_obs   = xobs[:,index,:,:]
            target_mask  = mask[:,index,:,:]
            target_mask_ = mask_[:,index,:,:]

        # gradient normalisation
        x = torch.Tensor.index_fill(x,1,torch_index,0)
        x = torch.Tensor.index_add(x,1,torch_index,target_x)
        xpred = self.model_AE(x)
        grad  = self.model_Grad.compute_Grad(target_x, xpred, target_obs, target_mask)
        if normgrad == 0. :
            _normgrad = torch.sqrt( torch.mean( grad**2 ) )
        else:
            _normgrad = normgrad
        for kk in range(0,self.NGrad):
            # gradient update
            ## Gradient-based minimization using a fixed-step descent
            if self.OptimType == 0:
                grad  = self.model_Grad( target_x, xpred, target_obs, target_mask , _normgrad )
            ## Gradient-based minimization using a CNN using a (sub)gradient as inputs
            elif self.OptimType == 1:
                if kk == 0:
                    grad  = self.model_Grad( target_x, xpred, target_obs, target_mask, g1 , _normgrad)
                else:
                    grad  = self.model_Grad( target_x, xpred, target_obs, target_mask, grad_old , _normgrad)
                grad_old = torch.mul(1.,grad)
            ## Gradient-based minimization using a LSTM using a (sub)gradient as inputs
            elif self.OptimType == 2:
                if kk == 0:
                    grad,hidden,cell  = self.model_Grad( target_x, xpred, target_obs, target_mask, g1, g2 , _normgrad )
                else:
                    grad,hidden,cell  = self.model_Grad( target_x, xpred, target_obs, target_mask, hidden, cell , _normgrad )
            # interpolation constraint
            if( self.InterpFlag == True ):
                # optimization update
                xnew = target_x - grad
                target_x = torch.add(torch.mul(target_x,target_mask), torch.mul(xnew,target_mask_))
            else:
                # optimization update
                target_x = torch.add(target_x,torch.mul(grad,-1.0))

        if self.OptimType == 1:
            return target_x,grad_old,_normgrad
        if self.OptimType == 2:
            return target_x,hidden,cell,_normgrad
        else:
            return target_x,_normgrad

